import duckdb
import os

mimic_path = "./data/mimic_db"
patients_file = os.path.join(mimic_path, "PATIENTS.csv")
diagnoses_file = os.path.join(mimic_path, "DIAGNOSES_ICD.csv")

print(f"Loading data from {mimic_path}...")

try:
    conn = duckdb.connect(":memory:")
    
    if os.path.exists(patients_file):
        print(f"Loading {patients_file}...")
        conn.execute(f"CREATE TABLE patients AS SELECT * FROM read_csv_auto('{patients_file}')")
        print("Patients table created.")
        print("Schema:")
        print(conn.execute("DESCRIBE patients").fetchdf())
    else:
        print(f"File not found: {patients_file}")

    if os.path.exists(diagnoses_file):
        print(f"Loading {diagnoses_file}...")
        conn.execute(f"CREATE TABLE diagnoses_icd AS SELECT * FROM read_csv_auto('{diagnoses_file}')")
        print("Diagnoses table created.")
        print("Schema:")
        print(conn.execute("DESCRIBE diagnoses_icd").fetchdf())
    else:
        print(f"File not found: {diagnoses_file}")

    print("\n--- Sample Queries ---")
    
    print("1. Count patients:")
    print(conn.execute("SELECT COUNT(*) FROM patients").fetchone())

    print("2. Sample patient:")
    print(conn.execute("SELECT * FROM patients LIMIT 1").fetchdf())

    print("3. Filter by DOB (checking date handling):")
    try:
        # Try a date comparison. If DOB is string, this might work lexicographically or fail if format differs.
        # DuckDB read_csv_auto usually infers dates if format is standard.
        print(conn.execute("SELECT COUNT(*) FROM patients WHERE DOB > '2050-01-01'").fetchone())
    except Exception as e:
        print(f"Date query failed: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
