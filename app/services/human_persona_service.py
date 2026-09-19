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
                "You are 'WardMitra AI' (वॉर्डमित्र), a warm, caring, highly knowledgeable and empathetic human-like "
                "civic assistant for Pimpri Chinchwad Municipal Corporation (PCMC), Pune, Maharashtra. "
                "Speak naturally like a real helpful municipal officer. Greet warmly with 'नमस्कार' (Hello), "
                "refer to yourself as 'वॉर्डमित्र' (WardMitra) and explain how PCMC can help resolve the citizen's civic issue or query. "
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
                "🙏 **नमस्कार! मी वॉर्डमित्र AI सहाय्यक आहे.**\n\n"
                f"आपण विचारलेल्या प्रश्नाबद्दल: *\"{text}\"*\n\n"
                "मी पिंपरी चिंचवड महानगरपालिकेतील सर्व नागरी सेवा, योजना, नियम आणि तक्रार निवारणासाठी सदैव तत्पर आहे.\n"
                "• 🕳️ **खड्डे किंवा रस्ते समस्या**: थेट फोटो पाठवून तक्रार नोंदवा\n"
                "• 💡 **पथदिवे व विद्युत समस्या**: खांब क्रमांक किंवा पत्ता सांगा\n"
                "• 🗑️ **कचरा व स्वच्छता**: घंटागाडी व कचराकुंडी निवारण\n"
                "• 🚰 **पाणीपुरवठा व ड्रेनेज**: जलद गळती दुरुस्ती व ड्रेनेज सफाई\n\n"
                "👉 *कृपया आपली नेमकी समस्या किंवा प्रश्न सांगा, मी त्वरित मार्गदर्शन करतो!*"
            )
        else:
            return (
                "🙏 **Hello! I am WardMitra AI, your dedicated PCMC Civic Assistant.**\n\n"
                f"Regarding your query: *\"{text}\"*\n\n"
                "I am here to assist you with municipal services, municipal procedures, and civic grievance resolution across PCMC.\n"
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
        detected_ward_name: Optional[str] = None
    ) -> str:
        """
        Generates empathetic, natural, human-like municipal officer dialogue
        acknowledging the citizen's civic problem, showing compassion, and guiding them to resolution.
        """
        ward_ctx = f" ({detected_ward_name})" if detected_ward_name else ""

        if has_substantive_desc:
            if lang == "mr":
                return (
                    f"नमस्कार! आपण सांगितलेली **{category_name}** संदर्भातील समस्या मी समजून घेतली आहे.{ward_ctx} नागरिकांना होणाऱ्या या गैरसोयीबद्दल आम्ही दिलगीर आहोत.\n\n"
                    f"मी ही तक्रार महापालिकेच्या संबंधित प्रभाग कक्षाकडे आणि कनिष्ठ अभियंत्यांकडे वर्ग करण्यासाठी तयार आहे.\n\n"
                    "👉 **आपली तक्रार अधिकृतपणे दाखल करायची आहे का?**\n"
                    "तक्रार नोंदवण्यासाठी आपण **'हो'** किंवा **'नोंदवा'** म्हणू शकता, किंवा समस्येचा फोटो (📷) जोडू शकता."
                )
            else:
                return (
                    f"Hello! I understand the issue you have reported regarding **{category_name}**.{ward_ctx} We sincerely regret the inconvenience this is causing.\n\n"
                    "I am ready to forward this matter directly to the assigned Ward Field Engineer for swift inspection and action.\n\n"
                    "👉 **Would you like me to officially register and submit this complaint?**\n"
                    "Please reply with **'Yes'** or **'Register'** to proceed, or attach a photo (📷)."
                )
        else:
            if lang == "mr":
                return (
                    f"नमस्कार! आपण **{category_name}** बाबत सांगत आहात. नागरिकांच्या तक्रारीचे वेळेत निवारण करणे हे आमचे कर्तव्य आहे.\n\n"
                    "या समस्येवर जलद कारवाई करण्यासाठी कृपया **नेमका रस्ता, चौक किंवा कॉलनीचे नाव** सांगू शकाल का? तसेच शक्य असल्यास एक **फोटो (📷)** जोडल्यास आमच्या महापालिका पथकाला जागेवर पोहोचणे अधिक सोपे होईल."
                )
            else:
                return (
                    f"Hello! I note that you are reporting an issue with **{category_name}**. We are committed to resolving municipal concerns promptly.\n\n"
                    "To help our field inspection team reach the exact spot, could you please mention the **specific street, chowk, or landmark**? Attaching a **photo (📷)** will also help us resolve it faster."
                )


human_persona_service = HumanPersonaService()
