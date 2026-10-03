"""
app.core.text_cleaning
~~~~~~~~~~~~~~~~~~~~~~
Sanitise LLM-generated markdown for voice (TTS) output.
No heavy imports — safe to use anywhere.
"""
import re

from app.core.chem_text import looks_like_smiles

# ---------------------------------------------------------------------------
# Compiled patterns (module-level — compiled once)
# ---------------------------------------------------------------------------

# Markdown fenced code blocks  ```...``` (may be multi-line)
_FENCED_CODE_RE   = re.compile(r"```[\s\S]*?```", re.MULTILINE)
# Inline code  `...`
_INLINE_CODE_RE   = re.compile(r"`[^`\n]*`")
# Markdown images  ![alt](url)
_MD_IMAGE_RE      = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Markdown links  [label](url) -> keep label
_MD_LINK_RE       = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Bare URLs (http / https / ftp)
_BARE_URL_RE      = re.compile(r"https?://\S+|ftp://\S+")
# Bold / italic markers: **, __, *, _
_BOLD_ITALIC_RE   = re.compile(r"\*{1,3}|_{1,3}")
# ATX headings: leading # symbols
_HEADING_RE       = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Blockquote markers: leading >
_BLOCKQUOTE_RE    = re.compile(r"^>\s*", re.MULTILINE)
# Bullet / unordered list markers: leading - * + (followed by space)
_BULLET_RE        = re.compile(r"^[\-\*\+]\s+", re.MULTILINE)
# Ordered list markers: "1. " style
_ORDERED_LIST_RE  = re.compile(r"^\d+\.\s+", re.MULTILINE)
# Table pipe rows (lines that start/end with |)
_TABLE_ROW_RE     = re.compile(r"^\|.*\|$", re.MULTILINE)
# Table separator rows (---|---...)
_TABLE_SEP_RE     = re.compile(r"^\|?[\s\-:]+\|[\s\-|:]+$", re.MULTILINE)
# Divider runs: ─── / --- / === / *** (3+ chars of the same glyph)
_DIVIDER_RE       = re.compile(r"[─\-=\*]{3,}")
# Emoji: broad Unicode blocks for emoticons, misc symbols, enclosed alphanumerics
#   U+1F300-U+1FAFF  (Miscellaneous Symbols and Pictographs, Emoji etc.)
#   U+2600-U+27BF    (Miscellaneous Symbols, Dingbats)
#   U+2460-U+2473    (Enclosed Alphanumerics: ①②…)
#   U+2776-U+277F    (Dingbat Negative Circled Digits: ❶❷…)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # emoticons, misc pictographs
    "\U00002600-\U000027BF"   # misc symbols / dingbats
    "\U00002460-\U00002473"   # enclosed alphanumerics ①…
    "\U00002776-\U0000277F"   # dingbat negative circled ❶…
    "]+",
    re.UNICODE,
)
# Multiple blank lines -> single blank line
_MULTI_BLANK_RE   = re.compile(r"\n{3,}")
# Leading/trailing whitespace on each line
_LINE_STRIP_RE    = re.compile(r"[ \t]+$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_for_tts(text: str) -> str:
    """
    Strip markdown and unreadable tokens from *text* so it can be spoken aloud.

    Operations (in order):
      1. Remove fenced code blocks (``` ... ```)
      2. Remove inline code (`...`)
      3. Remove markdown images (![alt](url))
      4. Convert markdown links [label](url) → label
      5. Remove bare URLs
      6. Remove ATX heading markers (# Title → Title)
      7. Remove blockquote markers (> ...)
      8. Remove bullet/ordered-list markers (- item → item; 1. item → item)
      9. Remove table rows and separators
     10. Remove divider runs (──── / ----)
     11. Remove bold/italic markers (** / __ / * / _)
     12. Remove emoji and circled-number glyphs
     13. Drop SMILES-like tokens
     14. Collapse whitespace / blank lines

    Idempotent: clean_for_tts(clean_for_tts(x)) == clean_for_tts(x)
    Returns "" if nothing speakable remains.
    """
    if not text:
        return ""

    t = text

    # 1. Fenced code blocks
    t = _FENCED_CODE_RE.sub(" ", t)
    # 2. Inline code
    t = _INLINE_CODE_RE.sub(" ", t)
    # 3. Markdown images
    t = _MD_IMAGE_RE.sub(" ", t)
    # 4. Markdown links → keep label
    t = _MD_LINK_RE.sub(r"\1", t)
    # 5. Bare URLs
    t = _BARE_URL_RE.sub(" ", t)
    # 6. Headings
    t = _HEADING_RE.sub("", t)
    # 7. Blockquotes
    t = _BLOCKQUOTE_RE.sub("", t)
    # 8. Bullet / ordered list markers
    t = _BULLET_RE.sub("", t)
    t = _ORDERED_LIST_RE.sub("", t)
    # 9. Table rows / separators
    t = _TABLE_ROW_RE.sub(" ", t)
    t = _TABLE_SEP_RE.sub(" ", t)
    # 10. Divider runs
    t = _DIVIDER_RE.sub(" ", t)
    # 11. Bold / italic markers
    t = _BOLD_ITALIC_RE.sub("", t)
    # 12. Emoji / circled numbers
    t = _EMOJI_RE.sub(" ", t)
    # 13. Drop SMILES-like tokens (word-split to preserve surrounding text)
    words = t.split()
    words = [w for w in words if not looks_like_smiles(w.strip(".,;:!?\"'"))]
    t = " ".join(words)
    # 14. Collapse whitespace
    t = _LINE_STRIP_RE.sub("", t)
    t = _MULTI_BLANK_RE.sub("\n\n", t)
    t = t.strip()

    return t
