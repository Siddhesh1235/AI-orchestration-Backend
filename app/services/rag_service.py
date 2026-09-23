"""
Ward Knowledge Base & RAG (Retrieval-Augmented Generation) Service.
Provides hyper-local, grounded civic knowledge retrieval for PCMC Sarathi / WardMitra AI.
Indexes Ward Handbooks (e.g. docs/WARD_15_HANDBOOK.md) and synthesizes accurate,
polite answers in Marathi and English with zero hallucination.
"""

import os
import re
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from app.clients.llm_client import llm_client

logger = logging.getLogger("pcms.rag_service")


@dataclass
class KnowledgeChunk:
    id: str
    ward_number: Optional[int]
    title: str
    content: str
    source_file: str
    tokens: List[str] = field(default_factory=list)
    vector: Optional[np.ndarray] = None


INQUIRY_INDICATORS = {
    "कधी", "कुठे", "कोण", "काय", "कसा", "कशी", "कसं", "कसे", "कस", "किती", "वेळ", "वेळापत्रक",
    "पत्ता", "कार्यालय", "नंबर", "संपर्क", "माहिती", "नियम", "दाखला", "सवलत",
    "रुग्णालय", "दवाखाना", "निरीक्षक", "अभियंता", "ऑफिस", "हॉस्पिटल",
    "what", "when", "where", "who", "which", "how", "timing", "schedule",
    "address", "officer", "inspector", "contact", "number", "helpline", "info", "details"
}


def is_knowledge_inquiry(text: str) -> bool:
    """Checks if text is asking an informational inquiry or question rather than reporting a complaint."""
    if not text:
        return False
    clean = text.lower().strip()
    words = set(re.findall(r'[\w\u0900-\u097F]+', clean))
    return bool(words & INQUIRY_INDICATORS) or "?" in clean


class WardDocumentLoader:
    """Loads and chunks Markdown Ward Handbooks into semantic knowledge units."""

    @staticmethod
    def load_from_directory(docs_dir: str = "docs") -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        base_path = Path(docs_dir)
        if not base_path.exists():
            # Check relative to project root
            base_path = Path(__file__).resolve().parent.parent.parent / "docs"

        if not base_path.exists():
            logger.warning(f"[RAG] Docs directory {base_path} not found.")
            return chunks

        # Filter strictly for ward handbooks and civic guides
        all_md = list(base_path.glob("*.md"))
        md_files = [
            f for f in all_md
            if "handbook" in f.name.lower() or "ward" in f.name.lower() or "मार्गदर्शक" in f.name.lower()
        ]
        if not md_files:
            md_files = all_md

        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                file_chunks = WardDocumentLoader._parse_markdown(content, md_file.name)
                chunks.extend(file_chunks)
                logger.info(f"[RAG] Loaded {len(file_chunks)} chunks from {md_file.name}")
            except Exception as e:
                logger.error(f"[RAG] Failed to read {md_file}: {e}")

        return chunks

    @staticmethod
    def _parse_markdown(content: str, filename: str) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []

        # Detect ward number from filename or content
        ward_num = None
        ward_match = re.search(r'(?:ward|प्रभाग)\s*(?:no\.?|cr\.?|क्र\.?|\b)?\s*(\d+)', filename, re.IGNORECASE)
        if not ward_match:
            ward_match = re.search(r'(?:ward|प्रभाग)\s*(?:no\.?|cr\.?|क्र\.?|\b)?\s*(\d+)', content[:300], re.IGNORECASE)
        if ward_match:
            ward_num = int(ward_match.group(1))

        # Split into sections based on ## headings
        sections = re.split(r'\n(?=##\s+)', content)
        chunk_idx = 0

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # Skip title banner preamble if it contains no substantive body
            if not sec.startswith("##") and len(sec) < 150:
                continue

            # Extract section title
            lines = sec.split("\n")
            title = "Ward Information"
            if lines[0].startswith("#"):
                title = lines[0].lstrip("#").strip()

            # If section contains sub-headings or tables or Q&As, split sensibly
            if "## ८. RAG सिस्टीमसाठी" in sec or "FAQs" in sec:
                # Split individual FAQs (प्र. १ / Q:)
                qa_blocks = re.split(r'\n(?=\*\*(?:प्र\.|Q))', sec)
                for qab in qa_blocks:
                    qab_clean = qab.strip()
                    if len(qab_clean) > 20:
                        qa_title = "Ward FAQ"
                        qa_first_line = qab_clean.split("\n")[0]
                        if "**" in qa_first_line:
                            qa_title = qa_first_line.replace("**", "").strip()
                        chunks.append(KnowledgeChunk(
                            id=f"{filename}_{chunk_idx}",
                            ward_number=ward_num,
                            title=qa_title,
                            content=qab_clean,
                            source_file=filename
                        ))
                        chunk_idx += 1
                continue

            # For regular sections, if section is very long (> 1200 chars), split into sub-blocks
            if len(sec) > 1200:
                subsections = re.split(r'\n(?=###\s+)', sec)
                for sub in subsections:
                    sub = sub.strip()
                    if not sub:
                        continue
                    sub_lines = sub.split("\n")
                    sub_title = title
                    if sub_lines[0].startswith("#"):
                        sub_title = f"{title} - {sub_lines[0].lstrip('#').strip()}"
                    chunks.append(KnowledgeChunk(
                        id=f"{filename}_{chunk_idx}",
                        ward_number=ward_num,
                        title=sub_title,
                        content=sub,
                        source_file=filename
                    ))
                    chunk_idx += 1
            else:
                chunks.append(KnowledgeChunk(
                    id=f"{filename}_{chunk_idx}",
                    ward_number=ward_num,
                    title=title,
                    content=sec,
                    source_file=filename
                ))
                chunk_idx += 1

        return chunks


class HybridVectorStore:
    """
    Lightweight, ultra-fast vector and semantic keyword store using numpy.
    Supports English, Marathi (Devanagari), and Hindi text without external C++ deps.
    """

    def __init__(self):
        self.chunks: List[KnowledgeChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.matrix: Optional[np.ndarray] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text into words and 3-char / 4-char n-grams for robust Indic morphological matching."""
        clean = text.lower().strip()
        # Word tokens (Marathi Devanagari + English alphanumeric)
        words = re.findall(r'[\w\u0900-\u097F]+', clean)
        tokens = list(words)
        # Add character 3-grams & 4-grams for words >= 4 chars to handle suffixes (उदा. 'मोरवाडीत' -> 'मोरवाडी')
        for w in words:
            if len(w) >= 4:
                for n in (3, 4):
                    for i in range(len(w) - n + 1):
                        tokens.append(w[i:i+n])
        return tokens

    def build_index(self, chunks: List[KnowledgeChunk]):
        """Builds TF-IDF vector matrix over chunks."""
        self.chunks = chunks
        if not chunks:
            logger.warning("[RAG] No chunks to index.")
            return

        # 1. Build vocabulary and term counts
        doc_freq: Dict[str, int] = {}
        all_doc_tokens: List[List[str]] = []

        for ch in chunks:
            text_to_index = f"{ch.title} {ch.title} {ch.content}"
            toks = self.tokenize(text_to_index)
            ch.tokens = toks
            all_doc_tokens.append(toks)
            unique_toks = set(toks)
            for t in unique_toks:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # Keep terms appearing at least once
        self.vocab = {term: idx for idx, term in enumerate(doc_freq.keys())}
        num_docs = len(chunks)
        vocab_size = len(self.vocab)

        # 2. Compute IDF vector
        idf_vec = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = doc_freq[term]
            idf_vec[idx] = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
        self.idf = idf_vec

        # 3. Compute normalized TF-IDF matrix
        matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
        for doc_idx, toks in enumerate(all_doc_tokens):
            term_counts: Dict[str, int] = {}
            for t in toks:
                term_counts[t] = term_counts.get(t, 0) + 1
            for t, cnt in term_counts.items():
                if t in self.vocab:
                    col_idx = self.vocab[t]
                    matrix[doc_idx, col_idx] = (cnt / len(toks)) * self.idf[col_idx]

            # L2 normalize
            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm

        self.matrix = matrix
        logger.info(f"[RAG] Built vector index for {num_docs} chunks. Vocabulary size: {vocab_size}")

    def search(
        self,
        query: str,
        ward_number: Optional[int] = None,
        top_k: int = 3,
        min_score: float = 0.12
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """Finds most relevant chunks using cosine similarity and keyword overlap."""
        if self.matrix is None or len(self.chunks) == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Vectorize query
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts: Dict[str, int] = {}
        for t in query_tokens:
            counts[t] = counts.get(t, 0) + 1

        for t, cnt in counts.items():
            if t in self.vocab:
                col_idx = self.vocab[t]
                q_vec[col_idx] = (cnt / len(query_tokens)) * self.idf[col_idx]

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        # Compute cosine similarity
        scores = np.dot(self.matrix, q_vec)

        # Keyword boost & ward filtering
        results: List[Tuple[KnowledgeChunk, float]] = []
        raw_query_lower = query.lower()

        for idx, chunk in enumerate(self.chunks):
            # Optional ward filter
            if ward_number and chunk.ward_number and chunk.ward_number != ward_number:
                continue

            sim_score = float(scores[idx])

            # Exact keyword boosting (e.g. specific officer names or services)
            title_lower = chunk.title.lower()
            content_lower = chunk.content.lower()

            words_in_q = [w for w in re.findall(r'[\w\u0900-\u097F]+', raw_query_lower) if len(w) >= 3]
            match_count = sum(1 for w in words_in_q if w in content_lower or w in title_lower)
            if words_in_q:
                keyword_boost = (match_count / len(words_in_q)) * 0.25
                sim_score += keyword_boost

            if sim_score >= min_score:
                results.append((chunk, sim_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class WardRAGService:
    """
    Orchestrates the Ward Knowledge Base RAG pipeline.
    Loads handbooks, retrieves relevant contexts, and generates grounded responses.
    """

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.store = HybridVectorStore()
        self._initialized = False
        self._ensure_initialized()

    def _ensure_initialized(self):
        if not self._initialized:
            chunks = WardDocumentLoader.load_from_directory(self.docs_dir)
            self.store.build_index(chunks)
            self._initialized = True

    def reindex(self):
        """Forces reload and re-indexing of all handbook files."""
        chunks = WardDocumentLoader.load_from_directory(self.docs_dir)
        self.store.build_index(chunks)
        logger.info(f"[RAG] Successfully re-indexed {len(chunks)} chunks.")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        ward_number: Optional[int] = 15,
        top_k: int = 3,
        min_score: float = 0.12
    ) -> List[Tuple[KnowledgeChunk, float]]:
        self._ensure_initialized()
        return self.store.search(query, ward_number=ward_number, top_k=top_k, min_score=min_score)

    def query_ward_knowledge(
        self,
        query: str,
        ward_number: Optional[int] = None,
        lang: str = "mr"
    ) -> Optional[str]:
        """
        End-to-end RAG query:
        1. Retrieves relevant ward handbook chunks.
        2. If matched, injects context into SmartLLMClient to generate a warm, grounded answer.
        3. Falls back gracefully if LLM is unavailable or confidence is low.
        """
        matches = self.retrieve(query, ward_number=ward_number, top_k=3, min_score=0.15)
        if not matches:
            return None

        best_chunk, best_score = matches[0]
        logger.info(f"[RAG] Matched chunk '{best_chunk.title}' with score {best_score:.3f} for query '{query[:50]}'")

        # Build clean context block from top matches
        context_blocks = []
        for ch, sc in matches:
            context_blocks.append(f"### {ch.title}\n{ch.content}")
        context_str = "\n\n".join(context_blocks)

        active_ward = best_chunk.ward_number or ward_number or 15

        # Generate grounded answer via LLM
        lang_name = "मराठी (Marathi)" if lang == "mr" else "English"
        system_prompt = (
            f"You are 'WardMitra AI' (वॉर्डमित्र), the friendly, knowledgeable ward digital assistant for the "
            f"WardMitra project (वॉर्डमित्र प्रकल्प), specializing in Ward {active_ward}.\n"
            "WardMitra is an independent ward-level civic assistant platform that helps citizens access local information "
            "and resolve civic problems in their ward.\n"
            f"Answer the citizen's question strictly based on the provided Ward {active_ward} Handbook context below.\n"
            "Rules:\n"
            "1. Use a warm, respectful tone (greet with 'नमस्कार').\n"
            "2. State the exact names, phone numbers, timings, addresses, or SLAs present in the context.\n"
            f"3. If details are not found in the context, politely state that they can contact the Ward {active_ward} Office or WardMitra support.\n"
            "4. Keep the answer clear, structured, and easy to read.\n"
            f"5. Answer in {lang_name}."
        )

        user_prompt = (
            f"Ward {active_ward} Handbook Context:\n"
            f"---------------------------------\n"
            f"{context_str}\n"
            f"---------------------------------\n\n"
            f"Citizen's Question: '{query}'\n"
            f"Helpful Official Answer in {lang_name}:"
        )

        try:
            answer = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
            if answer and len(answer.strip()) > 20:
                return answer.strip()
        except Exception as e:
            logger.warning(f"[RAG] LLM generation failed ({e}), falling back to direct context synthesis.")

        # Extractive fallback if LLM is unavailable
        is_multi_topic = any(conjn in query.lower() for conjn in ["आणि", "तसेच", "व", "and", "plus", "both"])
        query_words = [w for w in re.findall(r'[\w\u0900-\u097F]+', query.lower()) if len(w) >= 3]
        
        # Check if best_chunk alone misses key query terms
        best_content_lower = (best_chunk.title + " " + best_chunk.content).lower()
        missing_terms = [w for w in query_words if w not in best_content_lower]
        
        chunks_to_synthesize = [best_chunk]
        if (is_multi_topic or len(missing_terms) >= 2) and len(matches) > 1:
            for next_chunk, next_score in matches[1:]:
                next_content_lower = (next_chunk.title + " " + next_chunk.content).lower()
                if any(w in next_content_lower for w in missing_terms):
                    chunks_to_synthesize.append(next_chunk)
                    break
        
        return self._format_extractive_fallback(chunks_to_synthesize, lang, ward_num=active_ward)

    def _format_extractive_fallback(self, chunk_or_chunks: Union[KnowledgeChunk, List[KnowledgeChunk]], lang: str, ward_num: Optional[int] = None) -> str:
        """
        Fallback formatter that synthesizes key info into a natural, warm, human-like response
        without raw textbook headers (e.g. 'प्र. ४:') or robotic citations.
        Supports single chunk or multi-chunk synthesis for composite inquiries.
        """
        chunks = chunk_or_chunks if isinstance(chunk_or_chunks, list) else [chunk_or_chunks]
        if not chunks:
            return ""

        clean_answers = []
        for chunk in chunks:
            raw_text = chunk.content.strip()

            # 1. Check if chunk is an FAQ pair (Extract only the answer body)
            clean_answer = ""
            m_ans = re.search(r'\*\*(?:उ|A|Ans|Answer)\s*:\s*\*\*\s*(.*)', raw_text, re.DOTALL | re.IGNORECASE)
            if not m_ans:
                m_ans = re.search(r'(?:^|\n)(?:उ|A|Ans|Answer)\s*:\s*(.*)', raw_text, re.DOTALL | re.IGNORECASE)
            
            if m_ans:
                clean_answer = m_ans.group(1).strip()
            else:
                # Not a standard FAQ block: strip markdown headers and question labels
                lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
                substantive_lines = []
                for line in lines:
                    if line.startswith("#") or re.match(r'^\*\*(?:प्र\.|Q\.|Question)', line, re.IGNORECASE):
                        continue
                    substantive_lines.append(line)
                clean_answer = "\n".join(substantive_lines[:10]).strip()

            # If answer starts with a greeting like 'नमस्कार!' remove duplicate greeting
            clean_answer = re.sub(r'^(?:नमस्कार!|नमस्कार\s*!|Hello!|Hello\s*!)\s*', '', clean_answer).strip()
            if clean_answer and clean_answer not in clean_answers:
                clean_answers.append(clean_answer)

        combined_answer = "\n\n".join(clean_answers)

        if lang == "mr":
            return (
                f"नमस्कार! 🙏\n\n"
                f"{combined_answer}\n\n"
                f"👉 *आपणास या समस्येबाबत अधिकृत तक्रार नोंदवायची असल्यास कृपया सांगा किंवा ठिकाण पाठवा, मी लगेच नोंदवून घेईन.*"
            )
        elif lang == "hi":
            return (
                f"नमस्ते! 🙏\n\n"
                f"{combined_answer}\n\n"
                f"👉 *यदि आप इस समस्या के लिए आधिकारिक शिकायत दर्ज करना चाहते हैं, तो कृपया बताएं या स्थान साझा करें।* "
            )
        else:
            return (
                f"Hello! 🙏\n\n"
                f"{combined_answer}\n\n"
                f"👉 *If you would like to register an official civic complaint for this, please let me know or share the location.*"
            )


# Global singleton instance
rag_service = WardRAGService()
