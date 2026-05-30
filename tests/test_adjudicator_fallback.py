import importlib
import duckdb

# Import the module under test
import scripts.execute_adjudication_agent as adjudicator


def test_rule_based_high_confidence():
    dossier = {"PVR_Percentage": 90, "Semantic_Context": "This entity is high-risk and flagged."}
    r = adjudicator.rule_based_adjudication(dossier)
    assert r["Verdict"] == "High Confidence SAR"
    assert r["SAR_Confidence_Score"] == 92


def test_rule_based_review_required():
    dossier = {"PVR_Percentage": 96, "Semantic_Context": ""}
    r = adjudicator.rule_based_adjudication(dossier)
    assert r["Verdict"] == "Review Required"
    assert r["SAR_Confidence_Score"] == 75


def test_rule_based_no_sar():
    dossier = {"PVR_Percentage": 50, "Semantic_Context": "low risk"}
    r = adjudicator.rule_based_adjudication(dossier)
    assert r["Verdict"] == "No SAR"
    assert r["SAR_Confidence_Score"] == 10


def test_evaluate_llm_fallback():
    dossier = {"PVR_Percentage": 90, "Semantic_Context": "high-risk"}
    # force-disable the LLM and ensure evaluate_dossier_with_llm delegates to the rule-based flow
    original_flag = adjudicator.OLLAMA_AVAILABLE
    adjudicator.OLLAMA_AVAILABLE = False
    try:
        expected = adjudicator.rule_based_adjudication(dossier, err_msg="test-fallback")
        result = adjudicator.evaluate_dossier_with_llm(dossier)
        assert result["Verdict"] == expected["Verdict"]
        assert result["SAR_Confidence_Score"] == expected["SAR_Confidence_Score"]
    finally:
        adjudicator.OLLAMA_AVAILABLE = original_flag


def test_validate_case_type_integrity_pass():
    con = duckdb.connect()
    con.execute("CREATE TABLE Adjudication_Results (Case_Type VARCHAR);")
    con.execute("INSERT INTO Adjudication_Results VALUES ('SUSPICIOUS_FRAUD'), ('BENIGN_BASELINE');")
    adjudicator.validate_case_type_integrity(con)
    con.close()


def test_validate_case_type_integrity_failure():
    con = duckdb.connect()
    con.execute("CREATE TABLE Adjudication_Results (Case_Type VARCHAR);")
    con.execute("INSERT INTO Adjudication_Results VALUES ('SUSPICIOUS_FRAUD'), (NULL);")
    try:
        adjudicator.validate_case_type_integrity(con)
        assert False, "validate_case_type_integrity should raise ValueError when Case_Type is NULL"
    except ValueError as exc:
        assert "Data quality violation" in str(exc)
    finally:
        con.close()
