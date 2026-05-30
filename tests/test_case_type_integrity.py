import duckdb
import scripts.execute_adjudication_agent as adjudicator


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
