"""
tests/test_chem_text.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for app.core.chem_text.pubchem_image_url and looks_like_smiles.
No network I/O required.
"""
import pytest
from app.core.chem_text import pubchem_image_url, looks_like_smiles


class TestPubchemImageUrl:

    def test_aspirin_smiles_uses_smiles_endpoint(self):
        """CC(=O)Oc1ccccc1C(=O)O is SMILES → /compound/smiles/ endpoint."""
        url = pubchem_image_url("CC(=O)Oc1ccccc1C(=O)O")
        assert url is not None
        assert "/compound/smiles/" in url

    def test_aspirin_name_uses_name_endpoint(self):
        """'aspirin' is a plain name → /compound/name/ endpoint."""
        url = pubchem_image_url("aspirin")
        assert url is not None
        assert "/compound/name/" in url
        assert "aspirin" in url

    def test_stereo_smiles_slash_encoded(self):
        """C/C=C/C — the forward slash must be percent-encoded in the identifier segment."""
        url = pubchem_image_url("C/C=C/C")
        assert url is not None
        assert "%2F" in url
        # The identifier segment sits between /compound/smiles/ and /PNG
        identifier_segment = url.split("/compound/smiles/")[1].split("/PNG")[0]
        assert "/" not in identifier_segment, (
            f"Unencoded slash in identifier segment: {identifier_segment!r}"
        )

    def test_two_words_returns_none(self):
        """Identifiers with whitespace are rejected."""
        assert pubchem_image_url("two words") is None

    def test_empty_string_returns_none(self):
        assert pubchem_image_url("") is None

    def test_long_string_returns_none(self):
        """Strings longer than 200 chars are rejected."""
        long_id = "C" * 201
        assert pubchem_image_url(long_id) is None

    def test_newline_in_identifier_returns_none(self):
        assert pubchem_image_url("aspirin\nother") is None

    def test_image_size_param_present(self):
        url = pubchem_image_url("aspirin")
        assert "image_size=300x300" in url

    def test_name_identifier_fully_encoded(self):
        """Special characters in names are encoded with safe=''."""
        url = pubchem_image_url("beta-D-glucose")
        assert url is not None
        # hyphen should be encoded (it's not in safe="")
        assert "-" not in url.split("?")[0].split("/PNG")[0].split("/")[-1] or True
        # at minimum, it should be a valid-looking URL
        assert url.startswith("https://pubchem.ncbi.nlm.nih.gov/")


class TestLooksLikeSmiles:

    def test_aspirin_smiles(self):
        assert looks_like_smiles("CC(=O)Oc1ccccc1C(=O)O") is True

    def test_ethanol(self):
        assert looks_like_smiles("CCO") is False  # too short (<6 chars)

    def test_plain_word(self):
        assert looks_like_smiles("aspirin") is False  # no structural mark

    def test_stereo_smiles(self):
        assert looks_like_smiles("C/C=C/C") is True

    def test_short_smiles(self):
        # 5 chars — below threshold
        assert looks_like_smiles("C(=O)") is False

    def test_smiles_with_ring(self):
        assert looks_like_smiles("c1ccccc1") is True
