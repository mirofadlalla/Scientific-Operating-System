"""
app.core.greeting
~~~~~~~~~~~~~~~~~
Pure greeting / small-talk detection. No agent, LLM, or I/O imports.
"""
import re

# Original scientific list plus domain terms that prefix-regexes were swallowing.
_SCIENTIFIC_KEYWORDS_EN = (
    "compound", "drug", "disease", "molecule", "chemical", "smiles", "admet",
    "screening", "pathway", "protein", "target", "receptor", "ligand", "inhibitor",
    "biomarker", "clinical", "genome", "dna", "rna", "enzyme", "pharmacology",
    "antagonist", "agonist", "toxicity", "mechanism", "binding", "dose",
    "pharmacokinetic", "metabolite", "assay",
)
_SCIENTIFIC_KEYWORDS_AR = (
    "مركب", "دواء", "مرض", "بروتين", "جين", "مسار", "علاج", "دراسة", "تحليل",
)

_SCIENTIFIC_EN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _SCIENTIFIC_KEYWORDS_EN) + r")\b",
    re.IGNORECASE,
)
_SCIENTIFIC_AR_RE = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(kw) for kw in _SCIENTIFIC_KEYWORDS_AR) + r")(?=\s|$)",
)

_SMILES_TOKEN_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#:/\\.]+$")
_SMILES_MARK_RE = re.compile(r"[()=#\d]")

_PUNCT_TAIL = r"[\s!.,?،؟]*"

_EN_GREETING_RE = re.compile(
    r"^(?:"
    r"hello|hi|hey|howdy|greetings|"
    r"good\s+(?:morning|afternoon|evening|night|day)|"
    r"how\s+are\s+you|how'?s\s+it\s+going|how\s+is\s+it\s+going|"
    r"how'?s\s+things|how\s+are\s+things|what'?s\s+up|sup|yo|"
    r"nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you|good\s+to\s+see\s+you|"
    r"thanks|thank\s+you|thank\s+you\s+so\s+much|many\s+thanks|cheers|appreciate\s+it|"
    r"bye|goodbye|see\s+you|take\s+care|later|farewell|have\s+a\s+good\s+one|"
    r"what\s+can\s+you\s+do|what\s+do\s+you\s+do|who\s+are\s+you|what\s+are\s+you|"
    r"tell\s+me\s+about\s+yourself|"
    r"ok|okay|sure|cool|great|awesome|got\s+it|understood|sounds\s+good|perfect|nice|"
    r"yes|no|maybe|yep|nope|yeah|nah|"
    r"welcome|you'?re\s+welcome|\bnp\b|no\s+problem|no\s+worries|anytime|"
    r"sorry|excuse\s+me|my\s+bad|apologies|pardon"
    r")"
    r"(?:\s+(?:there|friend|everyone|all|again|folks))?"
    r"(?:\s+(?:so\s+much|a\s+lot))?"
    + _PUNCT_TAIL
    + r"$",
    re.IGNORECASE,
)

_HELP_RE = re.compile(
    r"^(?:help|i\s+need\s+help|can\s+you\s+help|can\s+you\s+assist)"
    r"(?:\s+\w+){0,3}"
    + _PUNCT_TAIL
    + r"$",
    re.IGNORECASE,
)

_AR_GREETING_PHRASES = (
    "السلام عليكم", "وعليكم السلام", "أهلاً", "أهلا", "مرحباً", "مرحبا", "هلا", "هلو", "هاي",
    "كيف حالك", "كيف الحال", "شلونك", "عامل إيه", "إيه أخبارك", "شنو أخبارك", "كيفك", "شو أخبارك",
    "صباح الخير", "صباح النور", "مساء الخير", "مساء النور", "تصبح على خير",
    "شكراً", "شكرا", "شكرًا", "اشكرك", "ممنون", "متشكر", "جزاك الله خيراً",
    "مع السلامة", "باي", "وداعاً", "في أمان الله", "إلى اللقاء", "يسلمك",
    "من أنت", "ما هو", "ماذا تفعل", "ماذا تعرف", "ما الذي يمكنك", "ايش تقدر تسوي",
    "نعم", "لا", "حسناً", "تمام", "موافق", "صحيح", "بالتأكيد", "ماشي", "اوكي",
    "آسف", "عذراً", "سامحني", "معليش", "مع احترامي",
    "سلام",
)
_AR_GREETING_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in _AR_GREETING_PHRASES) + r")"
    + _PUNCT_TAIL
    + r"$",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _has_smiles_token(text: str) -> bool:
    for token in text.split():
        token = token.strip(".,;:!?\"'()")
        if len(token) < 6:
            continue
        if _SMILES_TOKEN_RE.fullmatch(token) and _SMILES_MARK_RE.search(token):
            return True
    return False


def _has_scientific_keyword(text: str) -> bool:
    if _SCIENTIFIC_EN_RE.search(text) or _SCIENTIFIC_AR_RE.search(text):
        return True
    return _has_smiles_token(text)


def is_general_greeting(text: str) -> bool:
    """Return True for greetings / small-talk that should skip scientific routing."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _has_scientific_keyword(stripped):
        return False
    if _word_count(stripped) > 6:
        return False

    if _HELP_RE.match(stripped) and _word_count(stripped) <= 4:
        return True
    if _EN_GREETING_RE.match(stripped):
        return True
    if _AR_GREETING_RE.match(stripped):
        return True
    return False


def should_skip_orchestrator(text: str) -> bool:
    return is_general_greeting(text)
