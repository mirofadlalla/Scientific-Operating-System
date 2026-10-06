"""
app.core.chem_text
~~~~~~~~~~~~~~~~~~
PubChem image URL builder and the shared SMILES-token heuristic.
No heavy imports — safe to use anywhere.
"""
import re
import urllib.parse

# ---------------------------------------------------------------------------
# Shared SMILES heuristic
# ---------------------------------------------------------------------------

# Characters that are legal in a SMILES string
_SMILES_CHARS_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()/\\=#:.]+$")
# At least one of these "structural" marks must be present
_SMILES_MARK_RE  = re.compile(r"[()=#\[\]@/\\]|\d")


def looks_like_smiles(token: str) -> bool:
    """
    Return True when *token* looks like a SMILES string:
      - at least 6 characters long,
      - composed entirely of SMILES-legal characters, and
      - contains at least one structural mark ( ) = # [ ] @ / \\ or digit.
    """
    return (
        len(token) >= 6
        and bool(_SMILES_CHARS_RE.fullmatch(token))
        and bool(_SMILES_MARK_RE.search(token))
    )


# ---------------------------------------------------------------------------
# PubChem image URL
# ---------------------------------------------------------------------------

def pubchem_image_url(identifier: str) -> "str | None":
    """
    Build a PubChem PNG URL for *identifier*.

    Returns None when:
      - identifier is empty,
      - longer than 200 characters, or
      - contains whitespace or newlines.

    Uses the /compound/smiles/ endpoint for SMILES strings and
    /compound/name/ for plain compound names.
    All special characters (including /) are percent-encoded with
    safe="" so SMILES stereo-bond slashes survive the URL round-trip.
    """
    if not identifier:
        return None
    if len(identifier) > 200:
        return None
    if re.search(r"\s", identifier):
        return None

    q    = urllib.parse.quote(identifier, safe="")
    kind = "smiles" if looks_like_smiles(identifier) else "name"
    return (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{kind}/{q}"
        f"/PNG?image_size=300x300"
    )
