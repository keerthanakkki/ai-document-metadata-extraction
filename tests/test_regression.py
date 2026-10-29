"""
tests/test_regression.py
────────────────────────
Regression tests covering:
  1. Renewal notice interpretation under standard conversion convention (e.g. 'two months' notice' -> 60)
  2. Party name extraction excluding father's / spouse's names (e.g. 'Sri Vishal Bhardwaj S/O Charnel Singh' -> 'Sri Vishal Bhardwaj')
  3. Extraction service propagation of complete document text (not truncated to 500 preview characters)
  4. Correct handling of genuinely absent fields without fabrication or hallucination
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from app.schemas import RawLLMOutput, ExtractionStatus
from app.services.extraction_service import (
    clean_party_name,
    parse_renewal_notice_days,
    _raw_to_result,
    extract_from_bytes,
)
from app.services.llm_service import SYSTEM_PROMPT
from tests.conftest import make_minimal_docx


class TestRenewalNoticeConversionConvention:
    """Requirement: Notice periods in months convert as 1 month = 30 days."""

    def test_two_months_notice_phrase(self):
        assert parse_renewal_notice_days("two months notice") == 60

    def test_two_months_apostrophe(self):
        assert parse_renewal_notice_days("two months' notice") == 60

    def test_numeric_two_months(self):
        assert parse_renewal_notice_days("2 months notice") == 60

    def test_three_months_notice(self):
        assert parse_renewal_notice_days("three months notice") == 90

    def test_numeric_three_months(self):
        assert parse_renewal_notice_days("3 months") == 90

    def test_one_month_notice(self):
        assert parse_renewal_notice_days("one month notice") == 30

    def test_numeric_one_month(self):
        assert parse_renewal_notice_days("1 month") == 30

    def test_explicit_days_string(self):
        assert parse_renewal_notice_days("15 days") == 15
        assert parse_renewal_notice_days("60 days") == 60

    def test_integer_input(self):
        assert parse_renewal_notice_days(60) == 60
        assert parse_renewal_notice_days(15) == 15

    def test_numeric_string_input(self):
        assert parse_renewal_notice_days("60") == 60

    def test_absent_notice_returns_none(self):
        assert parse_renewal_notice_days(None) is None
        assert parse_renewal_notice_days("null") is None
        assert parse_renewal_notice_days("nan") is None
        assert parse_renewal_notice_days("") is None


class TestPartyNameParentageExclusion:
    """Requirement: Exclude father's, mother's, or spouse's names introduced by S/O, D/O, W/O."""

    def test_tenant_with_father_name_so(self):
        # Specific user-reported case
        raw = "Sri Vishal Bhardwaj S/O Charnel Singh"
        assert clean_party_name(raw) == "Sri Vishal Bhardwaj"

    def test_tenant_with_father_name_lowercase_so(self):
        raw = "Sri Vishal Bhardwaj s/o Charnel Singh"
        assert clean_party_name(raw) == "Sri Vishal Bhardwaj"

    def test_tenant_with_mixed_case_so(self):
        raw = "Mr.B.Kishore , S/o. Mr. B. Pampaiah"
        assert clean_party_name(raw) == "Mr.B.Kishore"

    def test_tenant_with_son_of_phrase(self):
        raw = "V.K.NATARAJ, son of V.KANDASWAMI CHETTIAR"
        assert clean_party_name(raw) == "V.K.NATARAJ"

    def test_tenant_with_daughter_of(self):
        raw = "Deepa Sharma, daughter of R.K. Sharma"
        assert clean_party_name(raw) == "Deepa Sharma"

    def test_tenant_with_wife_of(self):
        raw = "Mrs. S.Sakunthala W/O Late Sundaram"
        assert clean_party_name(raw) == "Mrs. S.Sakunthala"

    def test_clean_name_unmodified(self):
        assert clean_party_name("Sri Hanumaiah") == "Sri Hanumaiah"
        assert clean_party_name("Balaji.R") == "Balaji.R"
        assert clean_party_name("VYSHNAVI DAIRY SPECIALITIES Private Ltd") == "VYSHNAVI DAIRY SPECIALITIES Private Ltd"

    def test_empty_or_none_returns_none(self):
        assert clean_party_name(None) is None
        assert clean_party_name("") is None
        assert clean_party_name("null") is None


class TestFullDocumentTextPropagation:
    """Requirement: The extraction service must receive the complete text, not truncated to UI preview limit."""

    def test_complete_text_reaches_call_llm(self):
        full_text = (
            "Preamble: Landlord Sri Hanumaiah and Tenant Sri Vishal Bhardwaj S/O Charnel Singh. "
            + ("Some middle clause text repeating terms. " * 50)
            + "Clause 8: The tenant shall vacate the premises on two months' notice. "
            + "Final schedule details and witness signatures at Bangalore."
        )
        assert len(full_text) > 1500  # Well beyond 500 characters

        mock_raw = RawLLMOutput(
            agreement_value="12000",
            agreement_start_date="01.04.2008",
            agreement_end_date="31.03.2009",
            renewal_notice_days=60,
            party_one="Sri Hanumaiah",
            party_two="Sri Vishal Bhardwaj",
            evidence={"renewal_notice_days": "Clause 8: two months' notice"},
            confidence={"renewal_notice_days": 0.95},
        )

        with patch("app.services.extraction_service.extract_text_from_docx", return_value=full_text), \
             patch("app.services.extraction_service.build_few_shot_examples", return_value=[]), \
             patch("app.services.extraction_service.call_llm", return_value=mock_raw) as mock_call_llm:

            res = extract_from_bytes(make_minimal_docx(), "test.docx")

            # Verify that call_llm was passed the ENTIRE text (not truncated to 500 chars)
            passed_doc_text = mock_call_llm.call_args[0][0]
            assert len(passed_doc_text) == len(full_text)
            assert "Clause 8: The tenant shall vacate the premises on two months' notice." in passed_doc_text

            # Verify that the 500-char limit only applies to document_text_preview
            assert len(res.document_text_preview) <= 500
            assert res.renewal_notice_days == 60
            assert res.party_two == "Sri Vishal Bhardwaj"


class TestGenuinelyAbsentFieldHandling:
    """Requirement: Missing fields return null without fabrication and are flagged for review."""

    def test_genuinely_absent_renewal_notice(self):
        mock_raw = RawLLMOutput(
            agreement_value="3000",
            agreement_start_date="20.09.2010",
            agreement_end_date="19.07.2011",
            renewal_notice_days=None,  # Not present in document (e.g. Maddireddy agreement)
            party_one="M.V.V. VIJAYA SHANKAR",
            party_two="MADDIREDDY BHARGAVA REDDY",
        )

        result = _raw_to_result(mock_raw, "Sample document without notice period.")

        assert result.renewal_notice_days is None
        assert result.field_details["renewal_notice_days"].value is None
        assert result.field_details["renewal_notice_days"].is_missing is True
        assert result.field_details["renewal_notice_days"].confidence_score == 0.0
        assert result.status == ExtractionStatus.PARTIAL


class TestSemanticPromptInstructions:
    """Requirement: Semantic extraction prompt clearly contains rules for full inspection, conversion and names."""

    def test_prompt_includes_full_inspection_rule(self):
        assert "COMPLETE DOCUMENT INSPECTION" in SYSTEM_PROMPT or "complete document" in SYSTEM_PROMPT.lower()

    def test_prompt_includes_two_months_notice_convention(self):
        assert "two months" in SYSTEM_PROMPT.lower()
        assert "60" in SYSTEM_PROMPT

    def test_prompt_includes_so_parentage_exclusion(self):
        assert "S/O" in SYSTEM_PROMPT
        assert "Charnel Singh" in SYSTEM_PROMPT or "father" in SYSTEM_PROMPT.lower()


class TestPartyTwoParentageExclusionExtended:
    """Requirement D: Exclude father's name and qualifiers (Slo, S.L., S/o) from tenant name."""

    def test_tenant_with_slo_qualifier(self):
        raw = "V. Ravi Kian,, Slo VR.G.Sastry"
        assert clean_party_name(raw) == "V. Ravi Kian"

    def test_tenant_with_sl_qualifier(self):
        raw = "V. Ravi Kian, S.L. V.R.G. Sastry"
        assert clean_party_name(raw) == "V. Ravi Kian"

    def test_tenant_with_so_dot_qualifier(self):
        raw = "Mr. V.V. Ravi Kian, S/o. VR.G.Sastry"
        assert clean_party_name(raw) == "Mr. V.V. Ravi Kian"


class TestPartyOneLessorRoleAndLocalityExclusion:
    """Requirement D: Fix party identification so lessor is selected from preamble, not property schedule locality."""

    def test_locality_name_resolved_to_actual_lessor(self):
        from app.services.extraction_service import resolve_party_one
        doc_text = """
        Rental agreement executed at Bangalore between:
        Mrs. S.Sakunthala, Hno: 4E,10-12-1, REDNAMALCAZAR, Vishakapatnam
        Hereinafter called as the LESSOR of the ONE PART;
        And: V. Ravi Kian of the OTHER PART.
        Schedule Property: situated at Main Road, Ram murthy nagar Bangalore-560037.
        """
        # Model mistook Ram Murthy (from Ram murthy nagar) for party_one
        resolved, ev = resolve_party_one("Ram Murthy", doc_text)
        assert resolved == "Mrs. S.Sakunthala"

    def test_valid_landlord_name_unaltered(self):
        from app.services.extraction_service import resolve_party_one
        doc_text = "between Sri Hanumaiah LESSOR and tenant."
        resolved, _ = resolve_party_one("Sri Hanumaiah", doc_text)
        assert resolved == "Sri Hanumaiah"


class TestCalendarMonthEndDateCalculation:
    """Requirement F: Calendar-month calculation for end date following assignment counting convention."""

    def test_eleven_month_lease_starting_first_of_april(self):
        from app.services.extraction_service import calculate_calendar_end_date
        # 01.04.2010 + 11-month lease concluding in March ends 31.03.2011 (March has 31 days)
        res = calculate_calendar_end_date("01.04.2010", duration_months=11)
        assert res == "31.03.2011"

    def test_off_by_one_march_thirty_corrected_to_thirty_one(self):
        from app.services.extraction_service import calculate_calendar_end_date
        # Model returned 30.03.2011 due to off-by-one; March has 31 days
        res = calculate_calendar_end_date("01.04.2010", raw_end_date="30.03.2011", duration_months=11)
        assert res == "31.03.2011"

    def test_mid_month_lease_day_minus_one(self):
        from app.services.extraction_service import calculate_calendar_end_date
        # 15.12.2012 + 11 months -> 14.11.2013 (test dataset convention)
        res = calculate_calendar_end_date("15.12.2012", duration_months=11)
        assert res == "14.11.2013"

    def test_july_seventh_mid_month_lease(self):
        from app.services.extraction_service import calculate_calendar_end_date
        # 07.07.2013 + 11 months -> 06.06.2014
        res = calculate_calendar_end_date("07.07.2013", duration_months=11)
        assert res == "06.06.2014"


class TestRentConflictDetectionAndReviewFlagging:
    """Requirement E: Retain both numeric and words amounts; flag conflict for review with low confidence."""

    def test_numeric_nine_thousand_words_nine_thousand_two_hundred_conflict(self):
        from app.services.extraction_service import resolve_agreement_value_and_conflict
        doc_text = (
            "RENT: The monthly rent payable by the Lessee to the LESSOR for the Schedule Property "
            "shall be Rs. 9,000/- (Nine thousand and two hundred rupees only). This amount shall be paid monthly. "
            "DEPOSIT: Security deposit of Rs. 80,000/- only."
        )
        val, conf, ev, has_conflict = resolve_agreement_value_and_conflict("9200", None, 0.95, doc_text)
        assert val == "9200"  # Prioritize written amount per project policy
        assert has_conflict is True
        assert conf <= 0.40
        assert "CONFLICT DETECTED" in ev
        assert "9,000" in ev
        assert "Nine thousand and two hundred" in ev

    def test_concordant_rent_has_no_conflict(self):
        from app.services.extraction_service import resolve_agreement_value_and_conflict
        doc_text = "RENT: monthly rent is Rs. 6500 (Six thousand five hundred rupees only)."
        val, conf, ev, has_conflict = resolve_agreement_value_and_conflict("6500", None, 0.95, doc_text)
        assert val == "6500"
        assert has_conflict is False
        assert conf >= 0.85


class TestRenewalNoticeOCRTypoAndFallback:
    """Requirement G: Notice period handles OCR typos ('on month') and fallback inspection of termination clauses."""

    def test_on_month_notice_ocr_typo(self):
        assert parse_renewal_notice_days("on month notice") == 30

    def test_termination_clause_fallback_in_doc_text(self):
        doc_text = "TERMNNA TION OF LEA SE: Either party may gIVE on month notice writing to terminate this ease agreement."
        assert parse_renewal_notice_days(None, doc_text=doc_text) == 30
