"""
app.core.orchestration
~~~~~~~~~~~~~~~~~~~~~~~
Central query routing and streaming logic.

Contains:
  - Greeting / small-talk detection
  - COMBINED_ORCHESTRATOR_PROMPT  (routing + domain classification in one LLM call)
  - COMPOSITE_DETECTION_PROMPT    (split multi-part questions)
  - split_composite_question()
  - route_and_stream()            (async generator — the heart of the system)
"""
import asyncio
import json
import logging
import re
import time
from typing import AsyncIterator

import app.core.state as state
from app import monitoring
from app.config import settings
from app.core.deps import (
    client,
    chemical_agent,
    medical_agent,
    rag_agent,
    orchestrator,
    short_memory,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Greeting / small-talk fast-path
# ──────────────────────────────────────────────────────────────────────────────

def is_general_greeting(text: str) -> bool:
    """Returns True for greetings, social messages, and casual questions that
    should skip the orchestrator and go straight to the friendly APP_AGENT."""
    text_lower = text.strip().lower()

    greeting_patterns = [
        r"^(hello|hi|hey|howdy|greetings|good\s*(morning|afternoon|evening|night|day)).*$",
        r"^(how are you|how('?s| is) it going|how('?s| are) things|what'?s up|sup|yo).*$",
        r"^(nice to meet you|pleased to meet you|good to see you).*$",
        r"^(thanks|thank you|thank you so much|many thanks|cheers|appreciate it).*$",
        r"^(bye|goodbye|see you|take care|later|farewell|have a good one).*$",
        r"^(what can you do|what do you do|who are you|what are you|tell me about yourself).*$",
        r"^(help|i need help|can you help|can you assist).*$",
        r"^(ok|okay|sure|cool|great|awesome|got it|understood|sounds good|perfect|nice).*$",
        r"^(yes|no|maybe|yep|nope|yeah|nah)$",
        r"^(welcome|you'?re welcome|np|no problem|no worries|anytime).*$",
        r"^(sorry|excuse me|my bad|apologies|pardon).*$",
        # Arabic
        r"^(السلام عليكم|وعليكم السلام|أهلاً|أهلا|مرحباً|مرحبا|هلا|هلو|هاي).*$",
        r"^(كيف حالك|كيف الحال|شلونك|عامل إيه|إيه أخبارك|شنو أخبارك|كيفك|شو أخبارك).*$",
        r"^(صباح الخير|صباح النور|مساء الخير|مساء النور|تصبح على خير).*$",
        r"^(شكراً|شكرا|شكرًا|اشكرك|ممنون|متشكر|جزاك الله خيراً).*$",
        r"^(مع السلامة|باي|وداعاً|في أمان الله|إلى اللقاء|يسلمك).*$",
        r"^(من أنت|ما هو|ماذا تفعل|ماذا تعرف|ما الذي يمكنك|ايش تقدر تسوي).*$",
        r"^(نعم|لا|حسناً|تمام|موافق|صحيح|بالتأكيد|ماشي|اوكي).*$",
        r"^(آسف|عذراً|سامحني|معليش|مع احترامي).*$",
        r"^(سلام)$",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True

    scientific_keywords = [
        "compound", "drug", "disease", "molecule", "chemical", "smiles", "admet",
        "screening", "pathway", "protein", "target", "receptor", "ligand", "inhibitor",
        "biomarker", "clinical", "genome", "dna", "rna", "enzyme", "pharmacology",
        "مركب", "دواء", "مرض", "بروتين", "جين", "مسار", "علاج", "دراسة", "تحليل",
    ]
    if len(text_lower.split()) <= 3 and not any(kw in text_lower for kw in scientific_keywords):
        return True

    return False


def should_skip_orchestrator(text: str) -> bool:
    return is_general_greeting(text)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Prompts
# ──────────────────────────────────────────────────────────────────────────────

COMBINED_ORCHESTRATOR_PROMPT = """\
You are the Central Brain of AI-lixir, an AI Scientific Operating System specializing in Drug Discovery.
Your tasks:
  1. Determine if the query is within domain.
  2. If within domain, route it to the correct agent.

Available agents:
  CHEMICAL_AGENT  → intents: CHEMICAL_SIMILARITY | ADMET_ANALYSIS | DRUG_REPURPOSING
      Use for: SMILES, chemical structures, ADMET properties, molecular similarity, virtual screening,
               questions about how ADMET works, MPNN models, CPU/GPU usage in chemistry pipelines,
               any technical question about the chemical analysis system.
  MEDICAL_AGENT   → intent: BIOMEDICAL_MECHANISM
      Use for: biological pathways, drug-target interactions, clinical reasoning, pharmacology,
               disease mechanisms, proteins, receptors, biomarkers, genomics, enzymes.
  RAG_AGENT       → intent: APP_SUPPORT_RAG
      Use for: questions about AI-lixir features, API docs, how-to guides, system documentation,
               "who built this", "who is your master/creator/owner", "what is AI-lixir",
               questions about the platform, its capabilities, or its team.
  APP_AGENT       → intent: APP_HELP
      Use for: greetings, casual chat, short replies, "who are you", "what can you do", thank-yous,
               any ambiguous message that does NOT clearly fit the scientific agents above.

CRITICAL ROUTING RULES:
  - Questions about HOW the system works technically → CHEMICAL_AGENT or MEDICAL_AGENT depending on context.
  - Questions about WHO built the system → RAG_AGENT (APP_SUPPORT_RAG).
  - Questions about drugs, molecules, diseases, biology, chemistry → ALWAYS route to scientific agents.
  - When the topic is REMOTELY related to drug discovery, cheminformatics, or biomedical science → NEVER OUT_OF_DOMAIN.
  - OUT_OF_DOMAIN is ONLY for topics with ZERO connection to science: pure law, cooking, sports, celebrity gossip.
  - When in doubt → APP_AGENT. NEVER reject science-adjacent questions.

Respond ONLY with a raw JSON object (no markdown, no explanation):
{
  "intent": "CHEMICAL_SIMILARITY"|"ADMET_ANALYSIS"|"DRUG_REPURPOSING"|"BIOMEDICAL_MECHANISM"|"APP_SUPPORT_RAG"|"APP_HELP"|"OUT_OF_DOMAIN",
  "target_agent": "CHEMICAL_AGENT"|"MEDICAL_AGENT"|"RAG_AGENT"|"APP_AGENT"|"NONE",
  "entities": {"compound": "", "smiles": "", "disease": ""},
  "out_of_domain_reason": "brief reason only when OUT_OF_DOMAIN, else empty string"
}
"""

COMPOSITE_DETECTION_PROMPT = """\
You decide whether a user message contains MULTIPLE DISTINCT questions that should be answered separately.

A COMPOSITE message has two or more clearly independent topics, e.g.:
  - "What is the ADMET of aspirin AND what are the diabetes pathways?"
  - "Analyse compound X, also tell me about COVID-19 drug targets"
  - "Compare ibuprofen and aspirin side effects, plus the SMILES of caffeine"

A SINGLE message is NOT composite even if it mentions several properties of the SAME subject:
  - "What are the ADMET properties and mechanism of aspirin?" → SINGLE
  - "How does metformin work and what are its side effects?" → SINGLE

IMPORTANT: Rewrite each sub-question as a complete, self-contained question (add context if needed).

Return ONLY valid JSON, no markdown fences:
{"is_composite": true,  "sub_questions": ["full question 1", "full question 2", ...]}
or
{"is_composite": false, "sub_questions": ["original question"]}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Composite question splitter
# ──────────────────────────────────────────────────────────────────────────────

async def split_composite_question(text: str) -> list[str]:
    """
    Returns a list of sub-questions when the input contains multiple distinct
    scientific questions. Returns [text] unchanged when it is a single question.
    """
    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": COMPOSITE_DETECTION_PROMPT},
                {"role": "user",   "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content)
        parts = [q.strip() for q in data.get("sub_questions", []) if q.strip()]
        if data.get("is_composite") and len(parts) > 1:
            logger.info(f"[Composite] Split into {len(parts)} sub-questions: {parts}")
            return parts
    except Exception as exc:
        logger.warning(f"[Composite] Detection failed: {exc} — treating as single question")
    return [text]


# ──────────────────────────────────────────────────────────────────────────────
# Core routing + streaming generator
# ──────────────────────────────────────────────────────────────────────────────

async def route_and_stream(
    text_input: str,
    session_id: str,
    user_id: str,
    allow_split: bool = True,
    store_memory: bool = True,
) -> AsyncIterator[str]:
    """
    Shared orchestration logic: routes a query through the expert agents and
    yields text tokens from the synthesis stream.

    Performance design:
      - Greetings: 1 LLM call (streaming response).
      - Scientific: 1 combined routing call + agent + synthesis = 3 calls total.
      - Out-of-domain: instant rejection from the combined routing output.
      - Composite: detected in 1 extra call, then each sub-question processed
        independently with memory written once for the original question.
    """

    # ── Fast path: greetings skip ALL routing ────────────────────────────────
    if should_skip_orchestrator(text_input):
        chat_history = short_memory.get_history(session_id, limit=5)
        messages = [
            {"role": "system", "content": (
                "You are AI-lixir, a friendly and knowledgeable AI Scientific Operating System "
                "specializing in Drug Discovery, Cheminformatics, and Biomedical Research. "
                "You were built by Omar Fadlallah, an AI Engineer and CS student at Mansoura University, Egypt. "
                "The user has sent a casual message, greeting, or conversational input. "
                "Respond warmly and helpfully in the SAME language the user used (Arabic or English). "
                "Be concise and natural. If it's a greeting, introduce yourself briefly and invite them "
                "to ask about drug discovery, molecular analysis, ADMET predictions, or biomedical topics. "
                "If asked who built you, who your master/creator/owner is: Omar Fadlallah. "
                "Never say you cannot help with greetings — always engage positively."
            )}
        ]
        for msg in chat_history:
            messages.append(msg)
        messages.append({"role": "user", "content": text_input})

        start_time = time.time()
        ttft_ms = None
        first_token_received = False
        token_count = 0

        stream = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        full_reply = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                if not first_token_received:
                    first_token_received = True
                    ttft_ms = (time.time() - start_time) * 1000
                token_count += 1
                full_reply += token
                yield token

        gen_duration = time.time() - start_time - ((ttft_ms or 0) / 1000)
        tps = (token_count / gen_duration) if gen_duration > 0 and token_count > 0 else 0.0

        monitoring.record_agent_call("APP_AGENT", "APP_HELP", 0, success=True)
        monitoring.record_tokens(
            model=settings.ORCHESTRATOR_MODEL,
            prompt_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
            completion_tokens=len(full_reply) // 4,
            ttft_ms=ttft_ms,
            tps=tps,
        )
        if store_memory:
            short_memory.add_message(session_id, "user", text_input)
            short_memory.add_message(session_id, "assistant", full_reply)
            if state.long_memory is not None:
                try:
                    state.long_memory.add_entry(
                        session_id,
                        f"User: {text_input}\nAssistant: {full_reply}",
                        metadata={"intent": "APP_HELP", "agent": "APP_AGENT"},
                    )
                except Exception as _lm_err:
                    logger.warning(f"long_memory.add_entry failed: {_lm_err}")
        return

    # ── Composite question detection ─────────────────────────────────────────
    if allow_split:
        sub_questions = await split_composite_question(text_input)
        if len(sub_questions) > 1:
            is_ar = bool(re.search(r"[\u0600-\u06FF]", text_input))
            header = (
                f"🔍 تم اكتشاف **{len(sub_questions)} أسئلة منفصلة**. سأجيب على كل واحدة:\n\n"
                if is_ar else
                f"🔍 Detected **{len(sub_questions)} sub-questions** — answering each:\n\n"
            )
            yield header
            combined_answer = header
            divider = "\n\n" + "─" * 52 + "\n"
            _nums = ["❶", "❷", "❸", "❹", "❺", "❻", "❼", "❽", "❾", "❿"]

            for idx, sub_q in enumerate(sub_questions, start=1):
                num = _nums[idx - 1] if idx <= 10 else f"{idx}."
                section_header = f"{divider}**{num} {sub_q}**\n\n"
                yield section_header
                combined_answer += section_header

                part_answer = ""
                async for token in route_and_stream(
                    sub_q, session_id, user_id,
                    allow_split=False,
                    store_memory=False,
                ):
                    yield token
                    part_answer += token

                combined_answer += part_answer + "\n"
                yield "\n"

            if store_memory:
                short_memory.add_message(session_id, "user", text_input)
                short_memory.add_message(session_id, "assistant", combined_answer)
                state.SESSION_MEMORY.setdefault(session_id, []).append({"role": "user",      "content": text_input})
                state.SESSION_MEMORY.setdefault(session_id, []).append({"role": "assistant", "content": combined_answer})
                if state.long_memory is not None:
                    try:
                        state.long_memory.add_entry(
                            session_id,
                            f"User: {text_input}\nAssistant: {combined_answer}",
                            metadata={"intent": "COMPOSITE", "agent": "MULTI"},
                        )
                    except Exception as _lm_err:
                        logger.warning(f"long_memory.add_entry failed: {_lm_err}")
            return

    # ── Combined routing + domain classification (single LLM call) ───────────
    chat_history = short_memory.get_history(session_id, limit=10)
    messages = [{"role": "system", "content": COMBINED_ORCHESTRATOR_PROMPT}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": text_input})

    target_agent = "APP_AGENT"
    intent       = "APP_HELP"
    entities     = {}
    out_of_domain_reason = ""

    try:
        response = await client.chat.completions.create(
            model=settings.ORCHESTRATOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        routing_output       = json.loads(response.choices[0].message.content)
        intent               = routing_output.get("intent", "APP_HELP")
        target_agent         = routing_output.get("target_agent", "APP_AGENT")
        entities             = routing_output.get("entities") or {}
        out_of_domain_reason = routing_output.get("out_of_domain_reason", "")
    except Exception as exc:
        logger.warning(f"Orchestrator routing failed: {exc}. Using fallback classifier.")
        classification = orchestrator.classify_intent(text_input)
        intent_raw = (classification.get("intent") or "").lower()
        entities   = classification.get("entities") or {}
        if intent_raw == "chemical":
            target_agent, intent = "CHEMICAL_AGENT", "CHEMICAL_SIMILARITY"
        elif intent_raw == "medical":
            target_agent, intent = "MEDICAL_AGENT", "BIOMEDICAL_MECHANISM"
        else:
            target_agent, intent = "APP_AGENT", "APP_HELP"

    # ── Out-of-domain ─────────────────────────────────────────────────────────
    if intent == "OUT_OF_DOMAIN" or target_agent == "NONE":
        monitoring.record_out_of_domain(out_of_domain_reason)
        is_ar = bool(re.search(r"[\u0600-\u06FF]", text_input))
        refusal = (
            "عذراً، هذا السؤال خارج نطاق تخصصي العلمي. 🧪\n\n"
            "أنا نظام تشغيل ذكاء اصطناعي علمي متخصص حصرياً في **اكتشاف الأدوية، التحليل الكيميائي، والآليات الطبية الحيوية**. "
            "لا يمكنني الإجابة على الأسئلة المتعلقة بالقوانين، المحاماة، الطب السريري الشخصي، أو أي مجالات عامة أخرى.\n\n"
            "**مجالات تخصصي تشمل:**\n"
            "1. 🧬 **النواة الحيوية**: دراسة المسارات البيولوجية، آليات الأمراض، والبروتينات المستهدفة.\n"
            "2. 🧪 **النواة الكيميائية**: البحث عن المركبات المتشابهة وتوقع الخصائص السمية والحيوية (SMILES & ADMET).\n"
            "3. 🤖 **منسق المهام العلمي**: تشغيل خطوط الفحص الافتراضي وإعادة توجيه الأدوية."
        ) if is_ar else (
            "I'm sorry, this query is outside my scientific domain. 🧪\n\n"
            "I am an AI Scientific OS specializing strictly in **Drug Discovery, Chemical Analysis, and Biomedical Mechanisms**. "
            "I cannot assist with topics like law, clinical medicine, general advice, or other unrelated fields.\n\n"
            "**My core capabilities include:**\n"
            "1. 🧬 **Bioinformatics Core**: Analyzing biological pathways, disease mechanisms, and target receptors.\n"
            "2. 🧪 **Cheminformatics Core**: Searching chemical similarity, predicting ADMET properties, and molecular analysis.\n"
            "3. 🤖 **Scientific Orchestration**: Running virtual screening pipelines for drug repurposing."
        )
        for word in refusal.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", refusal)
        return

    # ── Parallel agent calls ──────────────────────────────────────────────────
    tasks, task_mapping = [], []
    chemical_intents = {"CHEMICAL_SIMILARITY", "ADMET_ANALYSIS", "DRUG_REPURPOSING"}
    has_chemical_data = entities.get("smiles") or entities.get("compound")
    if (intent in chemical_intents or target_agent == "CHEMICAL_AGENT") and has_chemical_data:
        tasks.append(chemical_agent.run(intent, entities))
        task_mapping.append("CHEMICAL")
    medical_intents = {"BIOMEDICAL_MECHANISM", "DRUG_REPURPOSING"}
    if entities.get("disease") and (intent in medical_intents or target_agent == "MEDICAL_AGENT"):
        tasks.append(medical_agent.run(intent, entities))
        task_mapping.append("MEDICAL")

    chemical_output = ""
    medical_output  = ""
    rag_output      = ""

    # ── RAG agent ──────────────────────────────────────────────────────────────
    if intent == "APP_SUPPORT_RAG" or target_agent == "RAG_AGENT":
        try:
            rag_output = await rag_agent.run(text_input)
            no_info_phrases = [
                "does not contain information", "not in the documentation",
                "knowledge base is currently", "documentation does not",
                "cannot find", "no information",
            ]
            if any(p in rag_output.lower() for p in no_info_phrases):
                logger.info("[RAG] No relevant docs — falling back to APP_AGENT.")
                rag_output   = ""
                target_agent = "APP_AGENT"
                intent       = "APP_HELP"
        except Exception as exc:
            rag_output = f"[RAG Agent Error]: {exc}"

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                if task_mapping[idx] == "CHEMICAL":
                    chemical_output = f"[Chemical Agent Error]: {res}"
                else:
                    medical_output = f"[Medical Agent Error]: {res}"
            else:
                if task_mapping[idx] == "CHEMICAL":
                    chemical_output = str(res)
                else:
                    medical_output = str(res)

    # ── RAG direct stream (no re-synthesis needed) ────────────────────────────
    if rag_output:
        for word in rag_output.split(" "):
            yield word + " "
            await asyncio.sleep(0.008)
        if store_memory:
            short_memory.add_message(session_id, "user", text_input)
            short_memory.add_message(session_id, "assistant", rag_output)
            state.SESSION_MEMORY.setdefault(session_id, []).append({"role": "user",      "content": text_input})
            state.SESSION_MEMORY.setdefault(session_id, []).append({"role": "assistant", "content": rag_output})
            if state.long_memory is not None:
                try:
                    state.long_memory.add_entry(
                        session_id,
                        f"User: {text_input}\nAssistant: {rag_output}",
                        metadata={"intent": intent, "agent": "RAG_AGENT"},
                    )
                except Exception as _lm_err:
                    logger.warning(f"long_memory.add_entry failed: {_lm_err}")
        return

    # ── Synthesis via LLM ─────────────────────────────────────────────────────
    if chemical_output or medical_output:
        agent_raw_output = f"[Chem Data]: {chemical_output}\n[Bio Data]: {medical_output}".strip()
    else:
        agent_raw_output = "[App System Context]: Standard greeting or help request."

    chat_history = short_memory.get_history(session_id, limit=12)
    is_arabic = bool(re.search(r"[\u0600-\u06FF]", text_input))
    lang_instruction = "Respond in Arabic (Egyptian dialect is fine)." if is_arabic else "Respond in English."

    messages = [
        {"role": "system", "content": (
            "You are AI-lixir, a scientific AI OS assistant specializing in Drug Discovery, "
            "Cheminformatics, and Biomedical Research. "
            "You were built by Omar Fadlallah, an AI Engineer from Egypt. "
            "Your job: synthesize a professional, clear answer based on the retrieved lab data and conversation history. "
            f"{lang_instruction} "
            "Use the retrieved data to directly answer the user's question. "
            "If the question is about how the system works technically (CPU, GPU, models, architecture), "
            "explain it clearly based on what you know about the system's design. "
            "NEVER say the question is outside your domain if it relates to science, chemistry, biology, "
            "drug discovery, or how this AI system works. "
            "If asked who built you or who your master/creator is: Omar Fadlallah."
        )}
    ]
    for msg in chat_history:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f'User Input Question: "{text_input}"\nRetrieved Lab Data: "{agent_raw_output}"',
    })

    start_time = time.time()
    ttft_ms = None
    first_token_received = False
    token_count = 0

    stream = await client.chat.completions.create(
        model=settings.ORCHESTRATOR_MODEL,
        messages=messages,
        temperature=0.3,
        stream=True,
    )

    full_reply   = ""
    _agent_start = time.time()
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            if not first_token_received:
                first_token_received = True
                ttft_ms = (time.time() - start_time) * 1000
            token_count += 1
            full_reply += token
            yield token

    gen_duration = time.time() - start_time - ((ttft_ms or 0) / 1000)
    tps = (token_count / gen_duration) if gen_duration > 0 and token_count > 0 else 0.0

    monitoring.record_agent_call(
        agent=target_agent,
        intent=intent,
        latency_ms=round((time.time() - _agent_start) * 1000, 2),
        success=True,
    )
    monitoring.record_tokens(
        model=settings.ORCHESTRATOR_MODEL,
        prompt_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
        completion_tokens=len(full_reply) // 4,
        ttft_ms=ttft_ms,
        tps=tps,
    )

    if store_memory:
        short_memory.add_message(session_id, "user", text_input)
        short_memory.add_message(session_id, "assistant", full_reply)
        state.SESSION_MEMORY.setdefault(session_id, []).append({"role": "user",      "content": text_input})
        state.SESSION_MEMORY.setdefault(session_id, []).append({"role": "assistant", "content": full_reply})
        if state.long_memory is not None:
            try:
                state.long_memory.add_entry(
                    session_id,
                    f"User: {text_input}\nAssistant: {full_reply}",
                    metadata={"intent": intent, "agent": target_agent},
                )
            except Exception as _lm_err:
                logger.warning(f"long_memory.add_entry failed: {_lm_err}")
