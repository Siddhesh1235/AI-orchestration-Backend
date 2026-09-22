"""
Human-Like Municipal Assistant Service for PCMC Sarathi / Ward Mitra AI.
Provides:
1. Warm, empathetic, polite municipal officer persona (Marathi & English).
2. Deep PCMC Civic Knowledge Base (Birth/Death Certificates, Property Tax, Water Bills,
   Waste Schedules, Ward Offices, Helplines, Building Permissions).
3. Conversational Smalltalk & General Inquiries (Human-like reasoning, greetings, thank yous).
"""

import re
import logging
from typing import Dict, Any, Optional

from app.clients.llm_client import llm_client

logger = logging.getLogger("pcms.human_persona")


PCMC_KNOWLEDGE_BASE = {
    "birth_death_certificate": {
        "patterns": [
            r"जन्म\s*(?:व\s*मृत्यू)?\s*(?:दाखला|प्रमाणपत्र)",
            r"मृत्यू\s*दाखला",
            r"birth\s*certificate",
            r"death\s*certificate",
            r"जन्म\s*नोंद"
        ],
        "mr": (
            "🙏 **पिंपरी चिंचवड महानगरपालिकेत जन्म किंवा मृत्यू दाखला मिळवण्याची सोपी पद्धत:**\n\n"
            "१️⃣ **ऑनलाइन पद्धत (जलद व घरबसल्या):**\n"
            "   • PCMC च्या अधिकृत पोर्टलवर जा: [pcmcindia.gov.in](https://www.pcmcindia.gov.in)\n"
            "   • 'नागरी सेवा (Citizen Services)' ➔ 'जन्म/मृत्यू नोंदणी' पर्यायावर क्लिक करा.\n"
            "   • बाळाचे नाव, जन्मतारीख किंवा रुग्णालय निवडून डिजिटल स्वाक्षरी असलेला अधिकृत दाखला मोफत डाउनलोड करा.\n\n"
            "२️⃣ **ऑफलाइन पद्धत:**\n"
            "   • आपल्या नजीकच्या **प्रभाग क्षेत्रीय कार्यालयातील नागरी सुविधा केंद्रात (CFC)** जा.\n"
            "   • आवश्यक कागदपत्रे: रुग्णालयाचा जन्म रिपोर्ट आणि पालकांचे आधार कार्ड.\n\n"
            "👉 *आपणास याव्यतिरिक्त आणखी काही मदत हवी आहे का?*"
        ),
        "en": (
            "🙏 **How to obtain a Birth or Death Certificate in PCMC:**\n\n"
            "1️⃣ **Online Mode (Instant & Free):**\n"
            "   • Visit the official PCMC portal: [pcmcindia.gov.in](https://www.pcmcindia.gov.in)\n"
            "   • Navigate to 'Citizen Services' ➔ 'Birth & Death Registration'.\n"
            "   • Enter the child's name, date of birth, or hospital to download the digitally signed certificate.\n\n"
            "2️⃣ **Offline Mode:**\n"
            "   • Visit the Citizen Facilitation Center (CFC) at your nearest PCMC Ward Office.\n"
            "   • Required documents: Hospital discharge summary and parents' Aadhaar card.\n\n"
            "👉 *Can I help you with any other municipal service?*"
        )
    },
    "property_tax": {
        "patterns": [
            r"घरपट्टी|मालमत्ता\s*कर|घर\s*टॅक्स",
            r"property\s*tax|house\s*tax",
            r"टॅक्स\s*भरायचा|कर\s*कसा\s*भरायचा"
        ],
        "mr": (
            "🏠 **पिंपरी चिंचवड महानगरपालिका मालमत्ता कर (घरपट्टी) माहिती:**\n\n"
            "१️⃣ **ऑनलाइन कर भरणा (Online Payment):**\n"
            "   • PCMC कर भरणा पोर्टल: [ptax.pcmcindia.gov.in](https://ptax.pcmcindia.gov.in)\n"
            "   • आपला **मालमत्ता क्रमांक (Property Index No.)** प्रविष्ट करा.\n"
            "   • UPI, डेबिट कार्ड, किंवा नेट बँकिंगद्वारे थेट पेमेंट करा आणि त्वरित पावती मिळवा.\n\n"
            "२️⃣ **सवलत (Concession / Rebate):**\n"
            "   • पहिल्या तिमाहीत (३० जून पूर्वी) आगाऊ सामान्य कर भरल्यास ५% ते १०% पर्यंत विशेष सवलत मिळते.\n\n"
            "३️⃣ **ऑफलाइन भरणा:** सर्व प्रभाग क्षेत्रीय कार्यालयातील कॅश काऊंटरवर सकाळी १० ते दुपारी ४ या वेळेत स्वीकारला जातो."
        ),
        "en": (
            "🏠 **PCMC Property Tax Payment Information:**\n\n"
            "1️⃣ **Online Payment:**\n"
            "   • Portal: [ptax.pcmcindia.gov.in](https://ptax.pcmcindia.gov.in)\n"
            "   • Enter your **Property Index Number** to view your tax bill.\n"
            "   • Pay via UPI, Net Banking, or Debit Card and download the official receipt instantly.\n\n"
            "2️⃣ **Early Payment Rebates:**\n"
            "   • PCMC offers a 5% to 10% rebate on general tax for payments made before June 30.\n\n"
            "3️⃣ **Offline Counters:** Available at all PCMC Ward Citizen Facilitation Centers (CFC) from 10:00 AM to 4:00 PM."
        )
    },
    "water_supply_timings": {
        "patterns": [
            r"पाणी\s*(?:कधी\s*येणार|वेळ|पुरवठा|गळती|बिल)",
            r"water\s*(?:timing|supply|bill|schedule|pressure)",
            r"पाणीपट्टी"
        ],
        "mr": (
            "🚰 **PCMC पाणीपुरवठा वेळापत्रक व माहिती:**\n\n"
            "• **नियमित पुरवठा:** पिंपरी चिंचवड शहरात प्रभागनिहाय दिवसाआड (Alternate Day) किंवा सकाळ/संध्याकाळ ठरावीक वेळेत पाणीपुरवठा होतो.\n"
            "• **सकाळची वेळ:** सकाळी ६:०० ते ९:००\n"
            "• **संध्याकाळची वेळ:** सायंकाळी ५:०० ते ८:०० (विस्तारित भागात)\n"
            "• **पाणीपट्टी ऑनलाइन भरणा:** [pcmcindia.gov.in](https://www.pcmcindia.gov.in) वरील पाणीपुरवठा विभागात जाऊन ग्राहक क्रमांक टाका.\n\n"
            "👉 *जर आपल्या भागात पाणी येत नसेल किंवा पाईपलाईन फुटली असेल, तर मला सांगा किंवा 'पाणी गळती' फोटो जोडून थेट तक्रार नोंदवा!*"
        ),
        "en": (
            "🚰 **PCMC Water Supply Schedule & Information:**\n\n"
            "• **Schedule:** Water is supplied alternate-day or morning/evening based on your specific ward.\n"
            "• **Morning Hours:** 6:00 AM – 9:00 AM\n"
            "• **Evening Hours:** 5:00 PM – 8:00 PM (Selected areas)\n"
            "• **Water Bill Payment:** Accessible online via the PCMC official portal.\n\n"
            "👉 *If you are facing no water supply or a pipe burst, please let me know or share a photo to register a water grievance immediately!*"
        )
    },
    "garbage_van_schedule": {
        "patterns": [
            r"कच(?:रा|ऱ्या|ऱ्याची)?\s*(?:गाडी|वेळ|कधी|उचलणे)",
            r"कचऱ्याची\s*गाडी",
            r"गाडी\s*कधी\s*(?:येते|येणार)",
            r"garbage\s*(?:van|timing|schedule|truck|collection)",
            r"घंटागाडी"
        ],
        "mr": (
            "🗑️ **PCMC घंटागाडी (कचरा संकलन) वेळापत्रक:**\n\n"
            "• **दैनिक संकलन वेळ:** दररोज सकाळी ७:०० ते दुपारी १२:०० वाजेपर्यंत सोसायट्या व घराघरांतून कचरा गोळा केला जातो.\n"
            "• **कचरा वर्गीकरण नियम:**\n"
            "  🟢 ओला कचरा (हिरवी बादली - स्वयंपाकघरातील कचरा)\n"
            "  🔵 सुका कचरा (निळी बादली - प्लास्टिक, कागद, पुठ्ठा)\n"
            "  🔴 घातक कचरा (सॅनिटरी, काच, औषधे - वेगळ्या पिशवीत)\n\n"
            "👉 *आपल्या भागात घंटागाडी वेळेवर येत नसेल किंवा कचरा साचला असेल, तर मला सांगा मी त्वरित आरोग्य निरीक्षकाकडे तक्रार पाठवतो!*"
        ),
        "en": (
            "🗑️ **PCMC Door-to-Door Waste Collection Schedule:**\n\n"
            "• **Daily Timing:** Every morning from 7:00 AM to 12:00 PM in residential societies and neighborhoods.\n"
            "• **Waste Segregation Guidelines:**\n"
            "  🟢 Wet Waste (Green bin: organic/kitchen food waste)\n"
            "  🔵 Dry Waste (Blue bin: paper, plastic, dry cartons)\n"
            "  🔴 Hazardous Waste (Separate wrapping: sanitary pads, broken glass, batteries)\n\n"
            "👉 *If the garbage vehicle hasn't arrived or waste is piling up, please let me know to notify the Ward Sanitary Inspector immediately!*"
        )
    },
    "helpline_numbers": {
        "patterns": [
            r"हेल्पलाईन|फोन\s*नंबर|कंट्रोल\s*रूम|कॉल\s*सेंटर|संपर्क",
            r"helpline|phone\s*number|control\s*room|call\s*center|contact\s*number",
            r"आयुक्त|कमिशनर"
        ],
        "mr": (
            "📞 **पिंपरी चिंचवड महानगरपालिका महत्त्वाचे संपर्क व हेल्पलाईन:**\n\n"
            "• **मध्यवर्ती सारथी कॉल सेंटर:** `020-67333333` (सकाळी ८ ते रात्री १०)\n"
            "• **आपत्कालीन नियंत्रण कक्ष (Disaster Cell):** `020-27425511` / `020-27425512`\n"
            "• **अग्निशामक दल (Fire Brigade):** `101` किंवा `020-27423333`\n"
            "• **रुग्णवाहिका (Ambulance):** `108`\n"
            "• **महापालिका मुख्य कार्यालय:** पिंपरी चिंचवड महानगरपालिका भवन, मुंबई-पुणे रस्ता, पिंपरी - ४११०१८\n\n"
            "👉 *मी सदैव आपल्या सेवेत आहे, आपण कोणतीही समस्या येथे थेट नोंदवू शकता!*"
        ),
        "en": (
            "📞 **PCMC Important Helplines & Municipal Contacts:**\n\n"
            "• **Central Sarathi Helpline:** `020-67333333` (8:00 AM to 10:00 PM)\n"
            "• **Disaster Management Cell (24x7):** `020-27425511` / `020-27425512`\n"
            "• **Fire Emergency:** `101` or `020-27423333`\n"
            "• **Medical Ambulance:** `108`\n"
            "• **PCMC Main Headquarters:** Mumbai-Pune Highway, Pimpri, Pune - 411018\n\n"
            "👉 *I am here 24/7 to assist you. You can report civic grievances directly to me anytime!*"
        )
    },
    "hospitals_health": {
        "patterns": [
            r"दवाखाना|रुग्णालय|हॉस्पिटल|आरोग्य\s*केंद्र",
            r"hospital|clinic|health\s*center|dispensary",
            r"वायसीएम|ycm"
        ],
        "mr": (
            "🏥 **PCMC प्रमुख रुग्णालये व आरोग्य केंद्रे:**\n\n"
            "१️⃣ **वाय. सी. एम. रुग्णालय (YCM Hospital):**\n"
            "   • संत तुकाराम नगर, पिंपरी (२४x७ आपत्कालीन व ट्रॉमा केअर सेंटर).\n"
            "२️⃣ **जिजामाता रुग्णालय:** पिंपरी गाव.\n"
            "३️⃣ **नवीन भोसरी रुग्णालय:** दिघी रस्ता, भोसरी.\n"
            "४️⃣ **तालेरा रुग्णालय:** चिंचवड स्टेशन.\n"
            "५️⃣ **प्रभाग आरोग्य केंद्रे:** सर्व ३२ प्रभागांमध्ये मोफत ओपीडी (OPD) व बाल लसीकरण सेवा उपलब्ध आहेत."
        ),
        "en": (
            "🏥 **PCMC Major Hospitals & Health Centers:**\n\n"
            "1️⃣ **YCM Hospital (Yashwantrao Chavan Memorial Hospital):**\n"
            "   • Sant Tukaram Nagar, Pimpri (24x7 Emergency, Multi-specialty & Trauma Care).\n"
            "2️⃣ **Jijamata Hospital:** Pimpri Gaon.\n"
            "3️⃣ **New Bhosari Hospital:** Dighi Road, Bhosari.\n"
            "4️⃣ **Talera Hospital:** Chinchwad Station.\n"
            "5️⃣ **Ward Dispensaries:** Available across all 32 PCMC wards providing free consultations, essential medicines, and vaccinations."
        )
    },
    "ward_offices": {
        "patterns": [
            r"प्रभाग\s*(?:कार्यालय|ऑफिस|क्षेत्रीय)",
            r"ward\s*office|zonal\s*office|क्षेत्रीय\s*कार्यालय"
        ],
        "mr": (
            "🏛️ **PCMC चे ८ क्षेत्रीय प्रभाग कार्यालये (Zonal Offices):**\n\n"
            "• **अ प्रभाग (Zone A):** निगडी प्राधिकरण (प्रभाग १, २, १३, १४)\n"
            "• **ब प्रभाग (Zone B):** चिंचवड स्टेशन (प्रभाग १६, १७, १८, २२)\n"
            "• **क प्रभाग (Zone C):** भोसरी (प्रभाग ३, ४, ५, ६)\n"
            "• **ड प्रभाग (Zone D):** रहाटणी / काळेवाडी (प्रभाग २३, २४, २५, २६)\n"
            "• **इ प्रभाग (Zone E):** भोसरी इंद्रायणी नगर (प्रभाग ७, ८, ९, १०)\n"
            "• **फ प्रभाग (Zone F):** रुपीनगर / तळवडे (प्रभाग ११, १२, १५)\n"
            "• **ग प्रभाग (Zone G):** पिंपरी (प्रभाग १९, २०, २१, २७)\n"
            "• **ह प्रभाग (Zone H):** सांगवी / नवी सांगवी (प्रभाग २८, २९, ३०, ३१, ३२)\n\n"
            "👉 *आपला परिसर सांगा, मी आपल्या प्रभागाचे थेट नाव व क्षेत्रीय अधिकाऱ्यांचा संपर्क देतो!*"
        ),
        "en": (
            "🏛️ **PCMC 8 Zonal Ward Offices:**\n\n"
            "• **Zone A:** Nigdi Pradhikaran (Wards 1, 2, 13, 14)\n"
            "• **Zone B:** Chinchwad Station (Wards 16, 17, 18, 22)\n"
            "• **Zone C:** Bhosari (Wards 3, 4, 5, 6)\n"
            "• **Zone D:** Rahatani / Wakad (Wards 23, 24, 25, 26)\n"
            "• **Zone E:** Bhosari Indrayani Nagar (Wards 7, 8, 9, 10)\n"
            "• **Zone F:** Rupinagar / Talwade (Wards 11, 12, 15)\n"
            "• **Zone G:** Pimpri / Nehrunagar (Wards 19, 20, 21, 27)\n"
            "• **Zone H:** Sangvi / Pimple Gurav (Wards 28, 29, 30, 31, 32)\n\n"
            "👉 *Mention your area or landmark to get your exact Ward Officer details!*"
        )
    },
    "new_water_connection": {
        "patterns": [
            r"नवीन\s*पाणी\s*(?:connection|कनेक्शन|नळ)",
            r"नवीन\s*नळ\s*कनेक्शन",
            r"(?:पाणी|नळ)\s*कनेक्शन\s*(?:साठी|लागणारी)?\s*(?:कागदपत्रे|documents)",
            r"water\s*connection\s*(?:documents?|requirements?|apply)",
            r"new\s*water\s*connection"
        ],
        "mr": (
            "💧 **पिंपरी चिंचवड महानगरपालिका — नवीन नळ/पाणी कनेक्शनसाठी आवश्यक कागदपत्रे:**\n\n"
            "१️⃣ **मालमत्ता कर पावती:** चालू आर्थिक वर्षातील घरपट्टी भरलेली पावती.\n"
            "२️⃣ **मालकी हक्क पुरावा:** इंडेक्स २ (Index-2), ७/१२ उतारा किंवा खरेदीखत प्रत.\n"
            "३️⃣ **ओळखपत्र:** अर्जदाराचे आधार कार्ड किंवा मतदान ओळखपत्र.\n"
            "४️⃣ **अधिकृत प्लंबर अहवाल:** मनपा परवानाधारक प्लंबरचा अहवाल व नकाशा.\n"
            "५️⃣ **बांधकाम मंजुरी/भोगवटा प्रमाणपत्र:** अधिकृत सँक्शन प्लॅन किंवा पूर्णत्वाचा दाखला.\n"
            "६️⃣ **सोसायटी एनओसी:** अपार्टमेंट/सोसायटीसाठी चेअरमन-सेक्रेटरी यांचे संमतीपत्र.\n\n"
            "👉 *अर्ज पद्धत:* PCMC नागरी सुविधा केंद्रात (CFC) किंवा अधिकृत पोर्टलवर ऑनलाइन अर्ज सादर करता येतो."
        ),
        "en": (
            "💧 **PCMC New Water Connection — Required Documents Checklist:**\n\n"
            "1️⃣ **Property Tax Receipt:** Current financial year's paid property tax bill.\n"
            "2️⃣ **Ownership Proof:** Index-2, 7/12 extract, or registered sale deed.\n"
            "3️⃣ **Identity Proof:** Applicant's Aadhaar Card or Voter ID.\n"
            "4️⃣ **Licensed Plumber Certificate:** Report and plumbing diagram from a PCMC-licensed plumber.\n"
            "5️⃣ **Building Sanction / Completion Certificate:** Official sanctioned building plan or occupancy certificate.\n"
            "6️⃣ **Society NOC:** No-Objection Certificate from society chairman/secretary (if applicable).\n\n"
            "👉 *How to apply:* Submit the application at your nearest PCMC Ward Citizen Facilitation Center (CFC) or online at pcmcindia.gov.in."
        )
    }
}


class HumanPersonaService:
    """Provides human-like, polite, empathetic conversational intelligence for PCMC Sarathi."""

    def find_knowledge_answer(self, text: str, lang: str = "mr") -> Optional[str]:
        """Scans query for municipal service questions and returns human-crafted answer."""
        t_clean = text.lower().strip()
        for key, entry in PCMC_KNOWLEDGE_BASE.items():
            for pat in entry["patterns"]:
                if re.search(pat, t_clean):
                    logger.info(f"[HumanPersona] Matched PCMC Knowledge Base: {key} (lang={lang})")
                    return entry.get(lang, entry["en"])

        # Check Ward RAG Service for hyper-local ward knowledge (Ward Handbook)
        try:
            from app.services.rag_service import rag_service, is_knowledge_inquiry
            if is_knowledge_inquiry(t_clean):
                # Detect specific ward from text if present (e.g. Ward 32, Ward 15, Sangvi, Pimpri)
                ward_num = None
                w_match = re.search(r'\b(?:ward|प्रभाग)\s*#?\s*(\d{1,2})\b', t_clean)
                if w_match:
                    try:
                        ward_num = int(w_match.group(1))
                    except Exception:
                        pass
                elif any(loc in t_clean for loc in ["sangvi", "सांगवी", "dhore nagar", "ढोरे नगर", "jaymala", "जयमाला", "madhuban", "मधुबन", "sangam nagar", "संगम नगर"]):
                    ward_num = 32
                elif any(loc in t_clean for loc in ["morwadi", "मोरवाडी", "pimpri gaon", "पिंपरी गाव"]):
                    ward_num = 15

                rag_ans = rag_service.query_ward_knowledge(text, ward_number=ward_num, lang=lang)
                if rag_ans:
                    return rag_ans
        except Exception as e:
            logger.debug(f"[HumanPersona] RAG query notice: {e}")

        return None

    def generate_human_response(self, text: str, lang: str = "mr") -> str:
        """
        Generates empathetic, natural, human-like responses to any user question.
        Combines:
        1. PCMC Civic Knowledge Base (Direct accurate answers)
        2. LLM with Warm Human Persona (if Ollama/OpenAI active)
        3. Empathetic Conversational Fallback
        """
        # 1. Check local PCMC Knowledge Base first
        kb_answer = self.find_knowledge_answer(text, lang)
        if kb_answer:
            return kb_answer

        # 2. Try LLM with Warm Human Municipal Persona
        llm_answer = None
        try:
            lang_name = "Marathi (मराठी)" if lang == "mr" else "English"
            system_prompt = (
                "You are 'WardMitra AI' (वॉर्डमित्र), a warm, caring, highly knowledgeable and empathetic "
                "civic assistant for the WardMitra project. "
                "Speak naturally like a real helpful ward companion. Greet warmly with 'नमस्कार' (Hello), "
                "refer to yourself as 'वॉर्डमित्र' (WardMitra) and explain how WardMitra can help resolve the citizen's civic issue or query. "
                "Keep your answers concise, polite, complete, and easy to read (2-4 sentences). "
                f"Always respond in {lang_name}."
            )
            user_prompt = (
                f"Citizen message: '{text}'\n"
                f"Please answer this question helpfully, warmly, and accurately in {lang_name}. "
                "If it relates to civic issues (potholes, water, streetlights, garbage, drainage), explain how to get it resolved. "
                "If it is a general, procedural, or conversational question, answer it thoroughly like a friendly government officer."
            )
            llm_answer = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.warning(f"[HumanPersona] LLM generation error: {e}")

        if llm_answer and len(llm_answer.strip()) > 15:
            cleaned_ans = llm_answer.strip()
            # If answer was cut off at the very end, trim to last complete sentence
            last_punct = max(
                cleaned_ans.rfind('.'),
                cleaned_ans.rfind('!'),
                cleaned_ans.rfind('?'),
                cleaned_ans.rfind('।'),
                cleaned_ans.rfind('\n')
            )
            if last_punct > 25:
                cleaned_ans = cleaned_ans[:last_punct + 1].strip()
            return cleaned_ans

        # 3. High-Quality Empathetic Fallback
        if lang == "mr":
            return (
                "🙏 **नमस्कार! मी वॉर्डमित्र (WardMitra AI) सहाय्यक आहे.**\n\n"
                f"आपण विचारलेल्या प्रश्नाबद्दल: *\"{text}\"*\n\n"
                "मी वॉर्डमित्र प्रकल्पांतर्गत प्रभाग स्तरावरील सर्व नागरी सेवा, स्थानिक अधिकारी संपर्क आणि तक्रार निवारणासाठी सदैव तत्पर आहे.\n"
                "• 🕳️ **खड्डे किंवा रस्ते समस्या**: थेट फोटो पाठवून तक्रार नोंदवा\n"
                "• 💡 **पथदिवे व विद्युत समस्या**: खांब क्रमांक किंवा पत्ता सांगा\n"
                "• 🗑️ **कचरा व स्वच्छता**: घंटागाडी व कचराकुंडी निवारण\n"
                "• 🚰 **पाणीपुरवठा व ड्रेनेज**: जलद गळती दुरुस्ती व ड्रेनेज सफाई\n\n"
                "👉 *कृपया आपली नेमकी समस्या किंवा प्रश्न सांगा, मी त्वरित मदत करतो!*"
            )
        else:
            return (
                "🙏 **Hello! I am WardMitra AI, your dedicated ward civic companion.**\n\n"
                f"Regarding your query: *\"{text}\"*\n\n"
                "I am here under the WardMitra project to assist you with local ward services, field officers, and civic grievance resolution.\n"
                "• 🕳️ **Potholes & Roads**: Share a photo or location to dispatch road repair staff\n"
                "• 💡 **Streetlights & Power**: Report dark streets or damaged poles\n"
                "• 🗑️ **Garbage & Cleanliness**: Request waste removal or schedule tracking\n"
                "• 🚰 **Water Supply & Drainage**: Report burst pipes, low pressure, or blocked gutters\n\n"
                "👉 *Please tell me how I can best assist you with your civic need today!*"
            )

    def generate_grievance_dialogue(
        self,
        text: str,
        category: str,
        category_name: str,
        has_substantive_desc: bool,
        lang: str = "mr",
        detected_ward_name: Optional[str] = None,
        turn_seed: int = 0
    ) -> str:
        """
        Generates empathetic, natural, human-like municipal officer dialogue
        acknowledging the citizen's civic problem, showing compassion, and guiding them to resolution.
        Supports rotating persona variations via turn_seed.
        """
        ward_ctx = f" ({detected_ward_name})" if detected_ward_name else ""

        if has_substantive_desc:
            if lang == "mr":
                mr_options = [
                    (
                        f"नमस्कार! आपण सांगितलेली **{category_name}** संदर्भातील समस्या मी समजून घेतली आहे.{ward_ctx} नागरिकांना होणाऱ्या या गैरसोयीबद्दल आम्ही दिलगीर आहोत.\n\n"
                        f"मी ही तक्रार महापालिकेच्या संबंधित प्रभाग कक्षाकडे आणि कनिष्ठ अभियंत्यांकडे वर्ग करण्यासाठी तयार आहे.\n\n"
                        "👉 **आपली तक्रार अधिकृतपणे दाखल करायची आहे का?**\n"
                        "तक्रार नोंदवण्यासाठी आपण **'हो'** किंवा **'नोंदवा'** म्हणू शकता, किंवा समस्येचा फोटो (📷) जोडू शकता."
                    ),
                    (
                        f"सादर प्रणाम! आपल्या परिसरातील **{category_name}** समस्येची मी दखल घेतली आहे.{ward_ctx} या त्रासाबद्दल आम्ही दिलगीर आहोत.\n\n"
                        f"आपल्या समस्येचे तातडीने निवारण करण्यासाठी ही माहिती संबंधित प्रभाग अभियंत्यांकडे पाठवली जाईल.\n\n"
                        "👉 **ही तक्रार महापालिकेत अधिकृतरीत्या दाखल करायची का?**\n"
                        "कृपया पुढे जाण्यासाठी **'होय'** किंवा **'नोंदवा'** सांगा."
                    ),
                    (
                        f"नमस्कार! **{category_name}** बाबत आपण दिलेला तपशील प्राप्त झाला आहे.{ward_ctx} नागरिकांची सुरक्षितता व सुविधा ही आमची प्राथमिकता आहे.\n\n"
                        "सर्व आवश्यक बाबींची नोंद झाली असून तक्रार नोंदणीसाठी तयार आहे.\n\n"
                        "👉 **आपण ही तक्रार सिस्टीममध्ये नोंदवू इच्छिता का?**\n"
                        "होकार देण्यासाठी **'होय'** म्हणा किंवा फोटो जोडा."
                    )
                ]
                return mr_options[turn_seed % len(mr_options)]
            elif lang == "hi":
                hi_options = [
                    (
                        f"नमस्ते! आपने **{category_name}** के बारे में जो समस्या बताई है, उसे मैंने समझ लिया है।{ward_ctx} नागरिकों को हो रही असुविधा के लिए हमें खेद है।\n\n"
                        "मैं इस शिकायत को संबंधित वार्ड इंजीनियर को अग्रेषित करने के लिए तैयार हूँ।\n\n"
                        "👉 **क्या आप इस शिकायत को आधिकारिक तौर पर दर्ज करना चाहते हैं?**\n"
                        "आगे बढ़ने के लिए **'हाँ'** या **'दर्ज करें'** कहें।"
                    ),
                    (
                        f"नमस्कार! **{category_name}** से संबंधित विवरण प्राप्त हुआ है।{ward_ctx}\n\n"
                        "त्वरित निवारण के लिए यह जानकारी फील्ड टीम को भेजी जा सकती है।\n\n"
                        "👉 **क्या मैं यह शिकायत सिस्टम में दर्ज करूँ?**\n"
                        "कृपया **'हाँ'** कहकर पुष्टि करें।"
                    )
                ]
                return hi_options[turn_seed % len(hi_options)]
            else:
                en_options = [
                    (
                        f"Hello! I understand the issue you have reported regarding **{category_name}**.{ward_ctx} We sincerely regret the inconvenience this is causing.\n\n"
                        "I am ready to forward this matter directly to the assigned Ward Field Engineer for swift inspection and action.\n\n"
                        "👉 **Would you like me to officially register and submit this complaint?**\n"
                        "Please reply with **'Yes'** or **'Register'** to proceed, or attach a photo (📷)."
                    ),
                    (
                        f"Greetings! I have noted your report concerning **{category_name}**.{ward_ctx} We apologize for the difficulty this has caused in your area.\n\n"
                        "All necessary information has been gathered to dispatch the municipal response team.\n\n"
                        "👉 **Shall I officially log this grievance with PCMC?**\n"
                        "Simply reply **'Yes'** or **'Register'** to finalize."
                    ),
                    (
                        f"Hello! Thank you for alerting us about the **{category_name}** problem.{ward_ctx} Ensuring prompt municipal service is our highest priority.\n\n"
                        "I have prepared this complaint for assignment to your zonal field supervisor.\n\n"
                        "👉 **Would you like to confirm and submit this complaint now?**\n"
                        "Reply with **'Yes'** to confirm."
                    )
                ]
                return en_options[turn_seed % len(en_options)]
        else:
            if lang == "mr":
                mr_options = [
                    (
                        f"नमस्कार! आपण **{category_name}** बाबत सांगत आहात. नागरिकांच्या तक्रारीचे वेळेत निवारण करणे हे आमचे कर्तव्य आहे.\n\n"
                        "या समस्येवर जलद कारवाई करण्यासाठी कृपया **नेमका रस्ता, चौक किंवा कॉलनीचे नाव** सांगू शकाल का? तसेच शक्य असल्यास एक **फोटो (📷)** जोडल्यास आमच्या महापालिका पथकाला जागेवर पोहोचणे अधिक सोपे होईल."
                    ),
                    (
                        f"सादर प्रणाम! **{category_name}** समस्येची दखल घेतली आहे.\n\n"
                        "आमच्या क्षेत्रीय कर्मचाऱ्यांना प्रत्यक्ष जागेवर पोहोचता यावे यासाठी कृपया **जवळची खूण (लँडमार्क), रस्ता किंवा प्रभाग** सांगा. सोबत फोटो असल्यास तोही पाठवू शकता."
                    ),
                    (
                        f"नमस्कार! आपण **{category_name}** ची तक्रार नोंदवत आहात.\n\n"
                        "कारवाई वेगाने होण्यासाठी कृपया समस्येचे **अंदाजे ठिकाण किंवा परिसराचे नाव** सांगा."
                    )
                ]
                return mr_options[turn_seed % len(mr_options)]
            elif lang == "hi":
                hi_options = [
                    (
                        f"नमस्ते! आपने **{category_name}** की समस्या का उल्लेख किया है।\n\n"
                        "फील्ड टीम को मौके पर भेजने के लिए कृपया **सटीक सड़क, चौक या नजदीकी लैंडमार्क** बताएं। फोटो (📷) होने पर वह भी साझा कर सकते हैं।"
                    ),
                    (
                        f"नमस्कार! **{category_name}** की शिकायत दर्ज करने के लिए कृपया समस्या का **स्थान या पता** बताएं।"
                    )
                ]
                return hi_options[turn_seed % len(hi_options)]
            else:
                en_options = [
                    (
                        f"Hello! I note that you are reporting an issue with **{category_name}**. We are committed to resolving municipal concerns promptly.\n\n"
                        "To help our field inspection team reach the exact spot, could you please mention the **specific street, chowk, or landmark**? Attaching a **photo (📷)** will also help us resolve it faster."
                    ),
                    (
                        f"Greetings! I have noted your **{category_name}** inquiry.\n\n"
                        "To dispatch our municipal crew accurately, please provide the **approximate address, road name, or landmark**. You may also share a photo (📷)."
                    ),
                    (
                        f"Hello! We appreciate you bringing this **{category_name}** issue to PCMC's notice.\n\n"
                        "Could you please state the **exact street or area** where this is located so we can proceed with registration?"
                    )
                ]
                return en_options[turn_seed % len(en_options)]

    def generate_image_category_confirmation(
        self,
        category_name: str,
        lang: str = "mr",
        turn_seed: int = 0
    ) -> str:
        """
        Generates natural phrasing asking citizen to confirm photo-detected category.
        """
        if lang == "mr":
            options = [
                f"मी आपल्या फोटोवरून समस्या **{category_name}** म्हणून ओळखली आहे. ही माहिती बरोबर आहे का?",
                f"आपण पाठवलेल्या छायाचित्रावरून ही **{category_name}** संदर्भातील तक्रार दिसते. आपण या प्रवर्गासह पुढे जाऊ इच्छिता का?",
                f"फोटो तपासणीनुसार ही समस्या **{category_name}** या प्रवर्गात येते. हे योग्य आहे का?"
            ]
        elif lang == "hi":
            options = [
                f"मैंने आपकी तस्वीर से इसे **{category_name}** के रूप में पहचाना है। क्या यह सही है?",
                f"भेजे गए फ़ोटो के अनुसार यह **{category_name}** की समस्या प्रतीत होती है। क्या हम इसके साथ आगे बढ़ें?",
                f"फ़ोटो विश्लेषण द्वारा यह समस्या **{category_name}** श्रेणी में पाई गई है। क्या यह उचित है?"
            ]
        else:
            options = [
                f"I have identified this image as: **{category_name}**. Is this correct?",
                f"Based on the photo you shared, this appears to be a **{category_name}** issue. Would you like to proceed with this category?",
                f"Photo analysis detected **{category_name}**. Does this match the issue you are facing?"
            ]
        return options[turn_seed % len(options)]

    def generate_location_prompt(
        self,
        category_name: str,
        lang: str = "mr",
        ward_label: Optional[str] = None,
        turn_seed: int = 0
    ) -> str:
        """
        Generates natural empathetic prompt asking for location.
        """
        ward_ctx = f" ({ward_label})" if ward_label else ""
        if lang == "mr":
            options = [
                f"छान! **{category_name}** बाबत तक्रार निश्चित केली आहे.{ward_ctx} कृपया या समस्येचे अंदाजे ठिकाण, रस्त्याचे नाव किंवा जवळची खूण (लँडमार्क) सांगा.",
                f"समजले! आमच्या महापालिका पथकाला **{category_name}** समस्येच्या जागेवर पोहोचण्यासाठी कृपया नेमका रस्ता, चौक किंवा कॉलनीचे नाव सांगा.{ward_ctx}",
                f"**{category_name}** ची नोंद घेण्यासाठी, कृपया ही समस्या कोणत्या भागात, रस्त्यावर किंवा चौकात आहे ते नमूद करा.{ward_ctx}"
            ]
        elif lang == "hi":
            options = [
                f"बहुत अच्छा! **{category_name}** की पुष्टि हो गई है।{ward_ctx} कृपया इसका अनुमानित स्थान, सड़क का नाम या नजदीकी लैंडमार्क बताएं।",
                f"समझ गया! फील्ड टीम के लिए कृपया **{category_name}** का सही पता, मार्ग या चौक बताएं।{ward_ctx}"
            ]
        else:
            options = [
                f"Great! I have confirmed the complaint as **{category_name}**.{ward_ctx} Please provide the approximate location, street name, or nearest landmark.",
                f"Understood! To help our field team inspect this **{category_name}** issue promptly, please specify the exact road, chowk, or colony.{ward_ctx}",
                f"Got it. Please share the address, area, or landmark where this **{category_name}** problem is located.{ward_ctx}"
            ]
        return options[turn_seed % len(options)]

    def generate_registration_confirmation_prompt(
        self,
        category_name: str,
        location: str,
        lang: str = "mr",
        ward_label: Optional[str] = None,
        turn_seed: int = 0
    ) -> str:
        """
        Generates prompt asking citizen to confirm registration.
        """
        loc_str = location or "आपल्या परिसरातील" if lang == "mr" else (location or "your area")
        ward_ctx = f" ({ward_label})" if ward_label else ""
        if lang == "mr":
            options = [
                f"धन्यवाद! माझ्याकडे **{loc_str}** येथील **{category_name}** समस्येची सर्व आवश्यक माहिती नोंदवली आहे.{ward_ctx} मी ही तक्रार अधिकृतपणे दाखल करू का?",
                f"उत्तम! **{loc_str}** येथे **{category_name}** बाबतची तक्रार दाखल करण्यासाठी तयार आहे.{ward_ctx} आपण ही तक्रार महापालिकेकडे नोंदवण्यास संमती देता का?",
                f"सर्व तपशील प्राप्त झाले आहेत: **{category_name}** (ठिकाण: **{loc_str}**).{ward_ctx} कृपया तक्रार नोंदवण्यासाठी **'हो'** किंवा **'नोंदवा'** म्हणा."
            ]
        elif lang == "hi":
            options = [
                f"धन्यवाद! मेरे पास **{loc_str}** पर **{category_name}** समस्या का पूरा विवरण है।{ward_ctx} क्या मैं यह शिकायत अधिकृत रूप से दर्ज करूँ?",
                f"सभी विवरण तैयार हैं: **{loc_str}** में **{category_name}**।{ward_ctx} शिकायत दर्ज करने के लिए कृपया **'हाँ'** कहें।"
            ]
        else:
            options = [
                f"Thank you! I have noted all required details for your **{category_name}** issue at **{loc_str}**.{ward_ctx} Would you like me to officially register this complaint now?",
                f"Great! The complaint for **{category_name}** at **{loc_str}** is ready.{ward_ctx} May I proceed with official registration?",
                f"All details are set: **{category_name}** at **{loc_str}**.{ward_ctx} Please confirm with **'Yes'** or **'Register'** to submit your complaint."
            ]
        return options[turn_seed % len(options)]

    def generate_registration_success_message(
        self,
        ticket_id: str,
        category_name: str,
        ward_text: str,
        sla_hours: int,
        worker_name: str,
        worker_contact: str,
        lang: str = "mr",
        turn_seed: int = 0,
        loc_text: Optional[str] = None
    ) -> str:
        """
        Generates human-like registration success message.
        All factual fields (ticket_id, category_name, ward_text, sla_hours, worker_name, worker_contact)
        are 100% deterministic and preserved.
        """
        loc_line_mr = f"\n• **📍 GPS स्थान (Location):** {loc_text}" if loc_text else ""
        loc_line_en = f"\n• **📍 GPS Location:** {loc_text}" if loc_text else ""

        if lang == "mr":
            options = [
                (
                    f"🎉 **आपली तक्रार यशस्वीरित्या नोंदवली गेली आहे!**\n\n"
                    f"• **तिकीट क्र:** {ticket_id}\n"
                    f"• **श्रेणी:** {category_name}\n"
                    f"• **🏛️ प्रभाग (Ward):** {ward_text}{loc_line_mr}\n"
                    f"• **👷 नेमलेले क्षेत्रीय कामगार:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ निकालाची मुदत (SLA):** {sla_hours} तास\n\n"
                    f"कामगार लवकरच समस्येचे निवारण करतील."
                ),
                (
                    f"✅ **तक्रार अधिकृतपणे दाखल झाली आहे!**\n\n"
                    f"• **तिकीट क्र:** {ticket_id}\n"
                    f"• **श्रेणी:** {category_name}\n"
                    f"• **🏛️ प्रभाग (Ward):** {ward_text}{loc_line_mr}\n"
                    f"• **👷 नियुक्त कर्मचारी:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ निवारण कालावधी (SLA):** {sla_hours} तास\n\n"
                    f"आमची क्षेत्रीय टीम तात्काळ कार्यवाहीसाठी रवाना झाली आहे."
                ),
                (
                    f"🤝 **आपली तक्रार PCMC प्रणालीत नोंदवली गेली आहे!**\n\n"
                    f"• **तिकीट क्र:** {ticket_id}\n"
                    f"• **श्रेणी:** {category_name}\n"
                    f"• **🏛️ प्रभाग (Ward):** {ward_text}{loc_line_mr}\n"
                    f"• **👷 जबाबदार कर्मचारी:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ अपेक्षित वेळ (SLA):** {sla_hours} तास\n\n"
                    f"आपल्या प्रभागाला स्वच्छ आणि सुरक्षित ठेवण्यासाठी सहकार्य केल्याबद्दल धन्यवाद! 🙏"
                )
            ]
            return options[turn_seed % len(options)]
        elif lang == "hi":
            options = [
                (
                    f"🎉 **आपकी शिकायत सफलतापूर्वक दर्ज कर ली गई है!**\n\n"
                    f"• **शिकायत क्र (Ticket ID):** {ticket_id}\n"
                    f"• **श्रेणी:** {category_name}\n"
                    f"• **🏛️ वार्ड:** {ward_text}{loc_line_en}\n"
                    f"• **👷 नियुक्त कर्मचारी:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ निवारण समय (SLA):** {sla_hours} घंटे\n\n"
                    f"संबंधित कर्मचारी जल्द ही समस्या का निवारण करेंगे।"
                ),
                (
                    f"✅ **शिकायत PCMC में पंजीकृत हो गई है!**\n\n"
                    f"• **टिकट क्र:** {ticket_id}\n"
                    f"• **श्रेणी:** {category_name}\n"
                    f"• **🏛️ वार्ड:** {ward_text}{loc_line_en}\n"
                    f"• **👷 फील्ड ऑफिसर:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ अधिकतम समय (SLA):** {sla_hours} घंटे\n\n"
                    f"निरीक्षण दल को सूचित कर दिया गया है।"
                )
            ]
            return options[turn_seed % len(options)]
        else:
            options = [
                (
                    f"🎉 **Your complaint has been successfully registered!**\n\n"
                    f"• **Ticket ID:** {ticket_id}\n"
                    f"• **Category:** {category_name}\n"
                    f"• **🏛️ Ward:** {ward_text}{loc_line_en}\n"
                    f"• **👷 Assigned Worker:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ Resolution Deadline (SLA):** {sla_hours} Hours\n\n"
                    f"The assigned worker has been dispatched."
                ),
                (
                    f"✅ **Complaint Officially Logged with PCMC!**\n\n"
                    f"• **Ticket ID:** {ticket_id}\n"
                    f"• **Category:** {category_name}\n"
                    f"• **🏛️ Ward:** {ward_text}{loc_line_en}\n"
                    f"• **👷 Field Staff:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ Target SLA:** {sla_hours} Hours\n\n"
                    f"Our team has been notified and will initiate field inspection shortly."
                ),
                (
                    f"🤝 **Your grievance has been successfully submitted!**\n\n"
                    f"• **Ticket ID:** {ticket_id}\n"
                    f"• **Category:** {category_name}\n"
                    f"• **🏛️ Ward:** {ward_text}{loc_line_en}\n"
                    f"• **👷 Assigned Official:** {worker_name} ({worker_contact})\n"
                    f"• **⏱️ SLA Window:** {sla_hours} Hours\n\n"
                    f"Thank you for helping keep Pimpri Chinchwad safe and well-maintained!"
                )
            ]
            return options[turn_seed % len(options)]


human_persona_service = HumanPersonaService()
