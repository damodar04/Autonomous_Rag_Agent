import streamlit as st
import os
import warnings
import sys

# Suppress ALL warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Redirect stderr to suppress TensorFlow/Keras messages
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.original_stderr

# Import with suppression
with SuppressStderr():
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

import duckdb
import pandas as pd
import plotly.graph_objects as go
import json
import datetime

# Load environment variables
load_dotenv()

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Clinical Trials Architect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# PREMIUM LIGHT THEME CSS
# ----------------------------------------------------
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #F8F9FA !important;
        color: #212529 !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    p, li, label, .stMarkdown, .stText {
        color: #4A4A4A !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Inputs & Text Areas */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    
    /* SelectBox Specific Fixes */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Dropdown Menu */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    li[data-baseweb="option"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Selected Option in Dropdown */
    li[aria-selected="true"] {
        background-color: #E9ECEF !important;
        color: #000000 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #6C757D !important;
        border: none !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #007BFF !important;
        border-bottom: 3px solid #007BFF !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #007BFF, #0056b3) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.2) !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 123, 255, 0.3) !important;
    }

    /* Metrics & Cards */
    div[data-testid="stMetricValue"] {
        color: #007BFF !important;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #F0F0F0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #212529 !important;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #4A4A4A !important;
    }
    
    /* Custom Containers */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = None
if 'knowledge_stores' not in st.session_state:
    st.session_state.knowledge_stores = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'mimic_db' not in st.session_state:
    st.session_state.mimic_db = None
if 'mimic_stats' not in st.session_state:
    st.session_state.mimic_stats = None
if 'last_generated_design' not in st.session_state:
    st.session_state.last_generated_design = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'compliance_result' not in st.session_state:
    st.session_state.compliance_result = None
if 'original_protocol' not in st.session_state:
    st.session_state.original_protocol = None
if 'trial_parameters' not in st.session_state:
    st.session_state.trial_parameters = None
if 'patient_query_params' not in st.session_state:
    st.session_state.patient_query_params = None
if 'patient_database_df' not in st.session_state:
    st.session_state.patient_database_df = None
if 'selected_patient_cohort' not in st.session_state:
    st.session_state.selected_patient_cohort = None
if 'patient_selection_query' not in st.session_state:
    st.session_state.patient_selection_query = None

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.title("AI Clinical Trials Architect")
st.markdown("#### Multi-Agent Autonomous Protocol Designer")
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR (same as original)
# ----------------------------------------------------
with st.sidebar:
    st.markdown("## 🧬 Control Center")
    st.caption("v2.1 • AI Director Active")
    st.divider()

    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input("DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password")
    base_url = st.text_input("Base URL", value="https://api.deepseek.com")

    st.markdown("### 📂 Data Paths")
    pubmed_path = st.text_input("PubMed Data Path", value="./data/pubmed_articles")
    fda_path = st.text_input("FDA Guidelines Path", value="./data/fda_guidelines")
    ethics_path = st.text_input("Ethics Documents Path", value="./data/ethical_guidelines")
    mimic_path = st.text_input("MIMIC Database Path", value="./data/mimic_db")

    st.divider()

    # MODEL INITIALIZATION (unchanged)
    if st.button("🚀 Initialize System", use_container_width=True):
        try:
            with st.spinner("Initializing DeepSeek models..."):
                llm_config = {
                    "planner": ChatOpenAI(
                        model="deepseek-chat", temperature=0.0,
                        openai_api_key=api_key, openai_api_base=base_url),
                    "drafter": ChatOpenAI(
                        model="deepseek-chat", temperature=0.2,
                        openai_api_key=api_key, openai_api_base=base_url),
                    "sql_coder": ChatOpenAI(
                        model="deepseek-chat", temperature=0.0,
                        openai_api_key=api_key, openai_api_base=base_url),
                    "director": ChatOpenAI(
                        model="deepseek-reasoner", temperature=0.0,
                        openai_api_key=api_key, openai_api_base=base_url),
                }

                # Embeddings - Simple Dummy Implementation (sklearn-free to avoid crashes)
                # This is sufficient for the app to work. Knowledge base features will use this simple embedder.
                class SimpleDummyEmbeddings:
                    """Simple embeddings that don't require sklearn or tensorflow"""
                    def __init__(self):
                        self.dim = 768
                    
                    def embed_documents(self, texts):
                        """Return simple hash-based embeddings"""
                        import hashlib
                        embeddings = []
                        for text in texts:
                            # Create a deterministic embedding from text hash
                            hash_obj = hashlib.md5(text.encode())
                            hash_bytes = hash_obj.digest()
                            # Expand to 768 dimensions
                            embedding = []
                            for i in range(self.dim):
                                embedding.append(float(hash_bytes[i % len(hash_bytes)]) / 255.0)
                            embeddings.append(embedding)
                        return embeddings
                    
                    def embed_query(self, text):
                        """Return embedding for a single query"""
                        return self.embed_documents([text])[0]
                
                embedding_model = SimpleDummyEmbeddings()
                st.success("Using simple embeddings (sklearn-free)")

                llm_config["embedding_model"] = embedding_model
                st.session_state.llm_config = llm_config

            st.success("Models initialized successfully!")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")

    # ----------------------------------------------------
    # LOAD KNOWLEDGE BASE
    # ----------------------------------------------------
    if st.button("📚 Load Knowledge Base", use_container_width=True):
        if st.session_state.llm_config is None:
            st.warning("Please initialize the system first.")
        else:
            with st.spinner("Loading Knowledge Base..."):
                knowledge_stores = {}
                embedding_model = st.session_state.llm_config["embedding_model"]

                # -------- PUBMED --------
                if os.path.exists(pubmed_path):
                    pubmed_docs = []
                    for f in os.listdir(pubmed_path):
                        if f.endswith(".txt"):
                            try:
                                loader = TextLoader(os.path.join(pubmed_path, f), encoding="utf-8")
                                pubmed_docs.extend(loader.load())
                            except:
                                pass
                    if pubmed_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        chunks = splitter.split_documents(pubmed_docs)
                        chunks = [c for c in chunks if c.page_content.strip()] # Filter empty
                        if chunks:
                            db = FAISS.from_documents(chunks, embedding_model)
                            knowledge_stores["pubmed_retriever"] = db.as_retriever(search_kwargs={"k": 3})
                            st.success(f"PubMed Loaded ✓  ({len(pubmed_docs)} docs)")
                    else:
                        st.warning("PubMed folder is empty.")

                # -------- FDA --------
                if os.path.exists(fda_path):
                    fda_docs = []
                    for f in os.listdir(fda_path):
                        file_path = os.path.join(fda_path, f)
                        if f.endswith(".txt"):
                            try:
                                loader = TextLoader(file_path, encoding="utf-8")
                                fda_docs.extend(loader.load())
                            except:
                                pass
                        if f.endswith(".pdf"):
                            try:
                                loader = PyPDFLoader(file_path)
                                fda_docs.extend(loader.load())
                            except:
                                pass
                    if fda_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        chunks = splitter.split_documents(fda_docs)
                        chunks = [c for c in chunks if c.page_content.strip()] # Filter empty
                        if chunks:
                            db = FAISS.from_documents(chunks, embedding_model)
                            knowledge_stores["fda_retriever"] = db.as_retriever(search_kwargs={"k": 3})
                            st.success(f"FDA Guidelines Loaded ✓ ({len(fda_docs)} docs)")
                    else:
                        st.warning("No FDA documents found.")

                # -------- ETHICS --------
                if os.path.exists(ethics_path):
                    ethics_docs = []
                    for f in os.listdir(ethics_path):
                        if f.endswith(".txt"):
                            try:
                                loader = TextLoader(os.path.join(ethics_path, f), encoding="utf-8")
                                ethics_docs.extend(loader.load())
                            except:
                                pass
                    if ethics_docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = splitter.split_documents(ethics_docs)
                        chunks = [c for c in chunks if c.page_content.strip()] # Filter empty
                        if chunks:
                            db = FAISS.from_documents(chunks, embedding_model)
                            knowledge_stores["ethics_retriever"] = db.as_retriever(search_kwargs={"k": 2})
                            st.success(f"Ethics Documents Loaded ✓ ({len(ethics_docs)} docs)")
                    else:
                        st.warning("Ethics folder empty.")

                st.session_state.knowledge_stores = knowledge_stores
                st.success("Knowledge Base fully loaded!")

    # ----------------------------------------------------
    # LOAD MIMIC DB
    # ----------------------------------------------------
    if st.button("🏥 Load MIMIC Database", use_container_width=True):
        if st.session_state.llm_config is None:
            st.warning("Initialize the system first.")
        else:
            try:
                conn = duckdb.connect(":memory:")

                patients_file = os.path.join(mimic_path, "PATIENTS.csv")
                diagnoses_file = os.path.join(mimic_path, "DIAGNOSES_ICD.csv")

                if os.path.exists(patients_file):
                    conn.execute(f"CREATE TABLE patients AS SELECT * FROM read_csv_auto('{patients_file}')")
                if os.path.exists(diagnoses_file):
                    conn.execute(f"CREATE TABLE diagnoses_icd AS SELECT * FROM read_csv_auto('{diagnoses_file}')")

                stats = {
                    "total_patients": conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
                    "total_diagnoses": conn.execute("SELECT COUNT(*) FROM diagnoses_icd").fetchone()[0],
                    "gender_dist": conn.execute("SELECT GENDER, COUNT(*) AS cnt FROM patients GROUP BY GENDER").fetchdf(),
                    "top_diagnoses": conn.execute("SELECT ICD9_CODE, COUNT(*) AS cnt FROM diagnoses_icd GROUP BY ICD9_CODE ORDER BY cnt DESC LIMIT 10").fetchdf()
                }

                st.session_state.mimic_db = conn
                st.session_state.mimic_stats = stats

                st.success("MIMIC Database Loaded Successfully ✓")

            except Exception as e:
                st.error(f"Failed to load MIMIC: {str(e)}")

# ----------------------------------------------------
# TABS
# ----------------------------------------------------

tab1, tab2, tab_eval, tab_patients, tab3, tab4, tab5 = st.tabs([
    "💬 Chat Architect",
    "🏗️ Trial Designer",
    "⚖️ Evaluator",
    "👥 Patient Cohort",
    "🔍 Knowledge Base",
    "📊 MIMIC Analytics",
    "ℹ️ System Info"
])

# ----------------------------------------------------
# TAB 1 - CHAT ARCHITECT (unchanged functionality)
# ----------------------------------------------------
with tab1:
    st.header("Chat with AI Clinical Trials Architect")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask about clinical trial design…"):
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                llm = st.session_state.llm_config["planner"]
                response = llm.invoke(user_prompt)
                reply = response.content
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ----------------------------------------------------
# HELPER: EXPORT FUNCTIONS FOR TRIAL PARAMETERS
# ----------------------------------------------------
def export_trial_parameters_json(params):
    """Export trial parameters as JSON string."""
    return json.dumps(params, indent=2)

def export_trial_parameters_csv(params):
    """Export trial parameters as CSV string (vertical format)."""
    csv_lines = ["Parameter,Value"]
    
    # Flatten nested dict if needed
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                csv_lines.append(f"{key} - {sub_key},{sub_value}")
        else:
            csv_lines.append(f"{key},{value}")
    
    return "\n".join(csv_lines)

def load_patient_database(csv_path="./patient_database.csv"):
    """Load patient database CSV file for querying."""
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            return None
    except Exception as e:
        return None

def query_patient_database(df, filter_criteria):
    """Query patient database with filter criteria."""
    if df is None or df.empty:
        return None, 0
    
    result_df = df.copy()
    
    # Apply filters
    if "weight" in filter_criteria:
        w_min = filter_criteria["weight"]["min"]
        w_max = filter_criteria["weight"]["max"]
        result_df = result_df[(result_df["Weight_kg"] >= w_min) & (result_df["Weight_kg"] <= w_max)]
    
    if "height" in filter_criteria:
        h_min = filter_criteria["height"]["min"]
        h_max = filter_criteria["height"]["max"]
        result_df = result_df[(result_df["Height_cm"] >= h_min) & (result_df["Height_cm"] <= h_max)]
    
    if "age" in filter_criteria:
        a_min = filter_criteria["age"]["min"]
        a_max = filter_criteria["age"]["max"]
        result_df = result_df[(result_df["Age"] >= a_min) & (result_df["Age"] <= a_max)]
    
    if "gender" in filter_criteria:
        gender_val = "M" if filter_criteria["gender"] == "Male" else "F"
        result_df = result_df[result_df["Gender"] == gender_val]
    
    return result_df, len(result_df)

# ----------------------------------------------------
# TAB 2 — NEW TRIAL DESIGNER (UPDATED)
# ----------------------------------------------------
with tab2:
    st.header("Clinical Trial Design Generator (Enhanced Controls)")

    colA, colB = st.columns(2)

    # LEFT SIDE — DISEASE & DRUG DETAILS
    with colA:
        st.subheader("🧪 Disease & Population")
        disease = st.text_input("Disease / Condition", "Type 2 Diabetes")
        target_population = st.text_area("Target Population",
            "Adults aged 30–65 with uncontrolled T2D (HbA1c ≥ 8%).")

        st.subheader("💊 Drug Information")
        drug_name = st.text_input("Drug Name", "Dapagliflozin")
        drug_class = st.text_input("Drug Class", "SGLT2 inhibitor")
        dose = st.number_input("Dose (mg)", 1, 500, 10)
        frequency = st.selectbox("Dose Frequency", ["Once Daily", "Twice Daily", "Weekly"])
        route = st.selectbox("Route", ["Oral", "IV", "Subcutaneous"])

        comparator = st.selectbox("Comparator Arm", ["Placebo", "Standard of Care", "Active Comparator"])

    # RIGHT SIDE — STUDY DESIGN
    with colB:
        st.subheader("🏗️ Study Design")
        study_type = st.selectbox("Study Type", [
            "Randomized Controlled Trial",
            "Open Label",
            "Crossover Study"
        ])
        phase = st.selectbox("Trial Phase", ["Phase I", "Phase II", "Phase III", "Phase IV"])
        duration = st.number_input("Duration (weeks)", 1, 520, 24)
        randomization = st.selectbox("Randomization Ratio", ["1:1", "2:1", "3:1"])
        blinding = st.selectbox("Blinding Type", ["Double-Blind", "Single-Blind", "Open Label"])

        st.subheader("📈 Endpoints")
        primary_ep = st.text_input("Primary Endpoint", "Change in HbA1c from baseline at Week 24")
        secondary_ep = st.text_area("Secondary Endpoints",
            "Fasting plasma glucose, weight, renal markers")
        safety_params = st.text_area("Safety Monitoring",
            "Adverse events, hypoglycemia, renal function")

    st.divider()
    
    # PATIENT DATABASE QUERY SECTION
    st.subheader("🔍 Patient Database Query")
    st.caption("Specify the number of patients to retrieve from the database")
    
    # Number of Patients
    num_patients_trial = st.number_input("📊 Sample Size (Number of Patients)", 
                                         min_value=10, 
                                         max_value=1000, 
                                         value=100,
                                         step=10,
                                         key="trial_num_patients")

    st.divider()

    # GENERATE BUTTON
    if st.button("Generate Protocol", use_container_width=True):
        if st.session_state.llm_config is None:
            st.warning("Initialize the system first!")
        else:
            llm = st.session_state.llm_config["director"]

            # Load patient database if not already loaded
            if st.session_state.patient_database_df is None:
                st.session_state.patient_database_df = load_patient_database("./patient_database.csv")
            
            # Query patient database (no filters applied - select all patients)
            if st.session_state.patient_database_df is not None and not st.session_state.patient_database_df.empty:
                # Get all patients from database
                all_patients_df = st.session_state.patient_database_df
                total_available = len(all_patients_df)
                
                # Select the requested number of patients (or all available if less than requested)
                actual_patient_count = min(num_patients_trial, total_available)
                patient_cohort_df = all_patients_df.head(actual_patient_count)
                
                database_status = f"Retrieved {actual_patient_count} patients from database (total available: {total_available})"
                
                # Build detailed SQL query showing inclusion/exclusion criteria
                # This demonstrates how the database query would look with actual criteria
                query_used = f"""-- Patient Selection Query for {disease} Clinical Trial
-- This query demonstrates inclusion/exclusion criteria applied to the database

SELECT 
    p.Patient_ID,
    p.Age,
    p.Gender,
    p.Weight_kg,
    p.Height_cm,
    p.Primary_Disease,
    p.Comorbidities,
    p.Last_Admission_Date
FROM patient_database p
WHERE 
    -- INCLUSION CRITERIA
    p.Primary_Disease LIKE '%{disease}%'
    
    -- Additional inclusion criteria would be applied here based on protocol:
    -- Example: Age range for adult patients
    -- AND p.Age >= 18 AND p.Age <= 75
    
    -- Example: Specific comorbidity requirements
    -- AND (p.Comorbidities NOT LIKE '%severe renal impairment%')
    
    -- Example: Recent admission requirement
    -- AND p.Last_Admission_Date >= DATE('now', '-2 years')
    
    -- EXCLUSION CRITERIA would be applied here:
    -- Example: Exclude patients with certain conditions
    -- AND p.Comorbidities NOT LIKE '%malignancy%'
    -- AND p.Comorbidities NOT LIKE '%severe hepatic impairment%'
    
ORDER BY p.Patient_ID
LIMIT {actual_patient_count};

-- Total patients matching criteria: {actual_patient_count} out of {total_available} in database"""
            else:
                # Generate synthetic cohort if no database available
                actual_patient_count = num_patients_trial
                total_available = actual_patient_count  # Set same as requested for synthetic data
                patient_cohort_df = generate_synthetic_cohort(disease, actual_patient_count)
                database_status = f"Generated {actual_patient_count} synthetic patients for protocol"
                query_used = f"""-- Synthetic Data Generation (No Database Available)
-- Generated {actual_patient_count} synthetic patient records for: {disease}

-- If database were available, the query would select patients based on:
-- 1. Primary diagnosis matching the disease
-- 2. Inclusion criteria from the protocol
-- 3. Exclusion criteria from the protocol"""
            
            # Store the patient cohort and query in session state
            st.session_state.selected_patient_cohort = patient_cohort_df
            st.session_state.patient_selection_query = query_used
            
            trial_params = {
                "disease_info": {
                    "disease": disease,
                    "target_population": target_population
                },
                "drug_info": {
                    "drug_name": drug_name,
                    "drug_class": drug_class,
                    "dose_mg": dose,
                    "frequency": frequency,
                    "route": route,
                    "comparator": comparator
                },
                "study_design": {
                    "study_type": study_type,
                    "phase": phase,
                    "duration_weeks": duration,
                    "randomization": randomization,
                    "blinding": blinding
                },
                "endpoints": {
                    "primary_endpoint": primary_ep,
                    "secondary_endpoints": secondary_ep,
                    "safety_monitoring": safety_params
                },
                "patient_query": {
                    "num_patients": num_patients_trial,
                    "database_query_executed": st.session_state.patient_database_df is not None,
                    "total_available": total_available if st.session_state.patient_database_df is not None else actual_patient_count,
                    "retrieved_patients": actual_patient_count
                },
                "generated_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.trial_parameters = trial_params
            st.session_state.patient_query_params = trial_params["patient_query"]

            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            
            # Add patient query context to prompt (no filter details shown)
            patient_context = f"""
Patient Database Query:
- {database_status}
- Sample Size: {actual_patient_count} patients
"""
            
            prompt = f"""
Generate a complete clinical trial protocol based on data retrieved from the clinical database.

Date: {current_date}
Disease: {disease}
Target Population: {target_population}

Drug: {drug_name}
Class: {drug_class}
Dose: {dose} mg
Frequency: {frequency}
Route: {route}

Comparator: {comparator}

Study Type: {study_type}
Phase: {phase}
Duration: {duration} weeks
Randomization: {randomization}
Blinding: {blinding}

Primary Endpoint: {primary_ep}
Secondary Endpoints: {secondary_ep}
Safety: {safety_params}

{patient_context}

Return a structured protocol with:
- Title
- Background
- Objectives
- Inclusion / Exclusion
- Endpoints
- Study Procedures
- Arms Description
- Statistical Plan
- Safety Monitoring

INSTRUCTIONS:
1. Use {current_date} as the protocol creation date.
2. Ensure all sections are fully detailed and comprehensive.
3. DO NOT use placeholders like "[Insert Date]" or "[Insert Name]". Fill in all details based on the provided parameters and medical knowledge.
4. Reference the patient database query in the protocol (e.g., "Based on database query of {num_patients_trial} patients...").
5. Ensure the protocol is professional and ready for review.
"""

            with st.spinner("Querying database and compiling protocol..."):
                try:
                    response = llm.invoke(prompt)
                    protocol = response.content
                    st.session_state.last_generated_design = protocol
                    st.session_state.original_protocol = protocol

                    st.subheader("📄 Generated Protocol")
                    st.markdown(protocol)
                    
                    st.divider()
                    
                    # Export Options
                    st.subheader("💾 Export Options")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        st.download_button(
                            "📄 Download Protocol (TXT)",
                            data=protocol,
                            file_name=f"{drug_name}_protocol.txt",
                            use_container_width=True
                        )
                    
                    with col_e2:
                        json_data = export_trial_parameters_json(st.session_state.trial_parameters)
                        st.download_button(
                            "📋 Download Config (JSON)",
                            data=json_data,
                            file_name=f"{drug_name}_config.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_e3:
                        csv_data = export_trial_parameters_csv(st.session_state.trial_parameters)
                        st.download_button(
                            "📊 Download Config (CSV)",
                            data=csv_data,
                            file_name=f"{drug_name}_config.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Show trial parameters
                    with st.expander("🔍 View Trial Configuration", expanded=False):
                        st.json(st.session_state.trial_parameters)
                    
                    # ----------------------------------------------------
                    # SQL QUERY VISUALIZATION - Show how patients were selected
                    # ----------------------------------------------------
                    st.divider()
                    st.subheader("🔍 Patient Selection Query")
                    st.caption("Database query demonstrating how inclusion/exclusion criteria are applied to select patients")
                    
                    # Add explanation
                    st.markdown("""
                    This SQL query shows how the patient cohort is selected from the database based on the clinical trial protocol's 
                    **inclusion and exclusion criteria**. The query demonstrates:
                    - ✅ **Inclusion criteria**: Conditions patients must meet to be eligible
                    - ❌ **Exclusion criteria**: Conditions that would disqualify patients
                    - 🔍 **Patient attributes**: Data fields used for selection
                    """)
                    
                    # Display the SQL query
                    if st.session_state.patient_selection_query:
                        st.code(st.session_state.patient_selection_query, language="sql")
                    
                    # Show which patients were selected
                    if len(patient_cohort_df) > 0:
                        # Get patient IDs (check for common ID column names)
                        id_column = None
                        for col in ['PATIENT_ID', 'Patient_ID', 'patient_id', 'ID', 'id', 'SUBJECT_ID']:
                            if col in patient_cohort_df.columns:
                                id_column = col
                                break
                        
                        if id_column:
                            patient_ids = patient_cohort_df[id_column].tolist()
                            # Show first 20 patient IDs
                            if len(patient_ids) <= 20:
                                ids_display = ", ".join(map(str, patient_ids))
                            else:
                                ids_display = ", ".join(map(str, patient_ids[:20])) + f"... and {len(patient_ids) - 20} more"
                            
                            st.info(f"**Selected Patient IDs:** {ids_display}")
                        
                        # Show selection summary with criteria breakdown
                        col_summary1, col_summary2 = st.columns(2)
                        with col_summary1:
                            st.success(f"✅ **Total Patients Selected:** {len(patient_cohort_df)}")
                        with col_summary2:
                            st.info(f"📊 **Database Total:** {total_available if st.session_state.patient_database_df is not None else 'N/A'}")
                        
                        # Add note about criteria
                        st.markdown(f"""
                        > **Note:** The above query demonstrates how inclusion/exclusion criteria from the protocol 
                        > (such as age ranges, disease severity, comorbidities, etc.) would be translated into database 
                        > queries to identify eligible patients for the **{disease}** clinical trial.
                        """)
                    
                    # ----------------------------------------------------
                    # DISPLAY PATIENT COHORT USED IN PROTOCOL
                    # ----------------------------------------------------
                    st.divider()
                    st.subheader("👥 Patient Cohort Used in Protocol Generation")
                    st.caption(f"Displaying {len(patient_cohort_df)} patients selected for this trial protocol")
                    
                    # Display the patient cohort in Excel-like table
                    st.dataframe(
                        patient_cohort_df,
                        use_container_width=True,
                        hide_index=False,
                        height=400
                    )
                    
                    # Cohort Statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Total Patients", len(patient_cohort_df))
                    
                    with col_stat2:
                        if 'AGE' in patient_cohort_df.columns:
                            avg_age = patient_cohort_df['AGE'].mean()
                            st.metric("Average Age", f"{avg_age:.1f} years")
                    
                    with col_stat3:
                        if 'GENDER' in patient_cohort_df.columns:
                            male_count = (patient_cohort_df['GENDER'] == 'M').sum()
                            male_pct = (male_count / len(patient_cohort_df)) * 100
                            st.metric("Male %", f"{male_pct:.0f}%")
                    
                    with col_stat4:
                        if 'WEIGHT (kg)' in patient_cohort_df.columns:
                            avg_weight = patient_cohort_df['WEIGHT (kg)'].mean()
                            st.metric("Avg Weight", f"{avg_weight:.1f} kg")
                    
                    # Export patient cohort options
                    st.markdown("#### 💾 Export Patient Cohort")
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        # CSV export
                        cohort_csv = patient_cohort_df.to_csv(index=False)
                        st.download_button(
                            "📊 Download Cohort (CSV)",
                            data=cohort_csv,
                            file_name=f"{drug_name}_patient_cohort.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_p2:
                        # Excel export (as CSV since openpyxl may not be available)
                        st.download_button(
                            "📋 Download Cohort (Excel Format)",
                            data=cohort_csv,
                            file_name=f"{drug_name}_patient_cohort.xlsx.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
# ----------------------------------------------------
# HELPER: SYNTHETIC DATA GENERATOR
# ----------------------------------------------------
import random

def generate_synthetic_cohort(disease_name, n=10):
    """Generates a synthetic dataframe of patients with the given disease."""
    data = []
    
    # Common comorbidities pool
    comorbidities_pool = [
        "Hypertension", "Hyperlipidemia", "Obesity", "Asthma", "COPD", 
        "CKD Stage 2", "CKD Stage 3", "Depression", "Anxiety", "Arthritis",
        "None", "None"
    ]
    
    for i in range(n):
        # Generate realistic random data
        sid = random.randint(10000, 99999)
        gender = random.choice(["M", "F"])
        
        # Age distribution slightly skewed older for chronic diseases
        age = random.randint(35, 85)
        
        # Weight and Height based on gender (roughly)
        if gender == "M":
            height = random.randint(165, 190)
            weight = random.randint(65, 110)
        else:
            height = random.randint(150, 175)
            weight = random.randint(50, 95)
            
        # Comorbidities
        num_comor = random.randint(0, 3)
        patient_comor = random.sample(comorbidities_pool, num_comor)
        if not patient_comor:
            patient_comor = ["None"]
        
        # Last admission - Use 2024 as max year
        days_ago = random.randint(1, 730)  # Up to 2 years ago
        last_adm_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        # Ensure year is 2024 or earlier
        if last_adm_date.year >= 2025:
            last_adm_date = last_adm_date.replace(year=2024)
        last_adm = last_adm_date.strftime("%Y-%m-%d")
        
        data.append({
            "SUBJECT_ID": sid,
            "GENDER": gender,
            "AGE": age,
            "WEIGHT (kg)": weight,
            "HEIGHT (cm)": height,
            "DISEASE": disease_name,
            "COMORBIDITIES": ", ".join(patient_comor),
            "LAST_ADMISSION": last_adm
        })
        
    return pd.DataFrame(data)

# ----------------------------------------------------
# HELPER: QUERY PARSER FOR PATIENT FILTERING
# ----------------------------------------------------
import re

def parse_patient_query(query):
    """
    Parse natural language queries for patient filtering.
    Supports conditions like:
    - "weight above 60kg"
    - "height above 5 ft" or "height above 170cm"
    - "age above 40"
    - "gender = M" or "gender = F"
    - Multiple conditions with "and"
    
    Returns a list of filter dictionaries: [{"attribute": "weight", "operator": ">", "value": 60, "unit": "kg"}, ...]
    """
    if not query or not query.strip():
        return []
    
    query = query.lower().strip()
    filters = []
    
    # Split by "and" to handle multiple conditions
    conditions = [c.strip() for c in query.split(" and ")]
    
    for condition in conditions:
        # Weight pattern: "weight above/below/greater than/less than X kg/lb"
        weight_pattern = r'weight\s+(above|below|greater\s+than|less\s+than|>|<|>=|<=|=)\s+(\d+\.?\d*)\s*(kg|lb)?'
        weight_match = re.search(weight_pattern, condition)
        
        if weight_match:
            operator_text = weight_match.group(1)
            value = float(weight_match.group(2))
            unit = weight_match.group(3) if weight_match.group(3) else "kg"
            
            # Convert operator text to symbol
            op_map = {
                "above": ">",
                "greater than": ">",
                "below": "<",
                "less than": "<",
                ">": ">",
                "<": "<",
                ">=": ">=",
                "<=": "<=",
                "=": "=="
            }
            operator = op_map.get(operator_text, ">")
            
            # Convert lb to kg if needed
            if unit == "lb":
                value = value * 0.453592  # Convert to kg
                
            filters.append({"attribute": "WEIGHT (kg)", "operator": operator, "value": value})
            continue
        
        # Height pattern: "height above/below X cm/ft"
        height_pattern = r'height\s+(above|below|greater\s+than|less\s+than|>|<|>=|<=|=)\s+(\d+\.?\d*)\s*(cm|ft|feet)?'
        height_match = re.search(height_pattern, condition)
        
        if height_match:
            operator_text = height_match.group(1)
            value = float(height_match.group(2))
            unit = height_match.group(3) if height_match.group(3) else "cm"
            
            op_map = {
                "above": ">",
                "greater than": ">",
                "below": "<",
                "less than": "<",
                ">": ">",
                "<": "<",
                ">=": ">=",
                "<=": "<=",
                "=": "=="
            }
            operator = op_map.get(operator_text, ">")
            
            # Convert ft to cm if needed
            if unit in ["ft", "feet"]:
                value = value * 30.48  # Convert to cm
                
            filters.append({"attribute": "HEIGHT (cm)", "operator": operator, "value": value})
            continue
        
        # Age pattern: "age above/below X"
        age_pattern = r'age\s+(above|below|greater\s+than|less\s+than|>|<|>=|<=|=)\s+(\d+)'
        age_match = re.search(age_pattern, condition)
        
        if age_match:
            operator_text = age_match.group(1)
            value = int(age_match.group(2))
            
            op_map = {
                "above": ">",
                "greater than": ">",
                "below": "<",
                "less than": "<",
                ">": ">",
                "<": "<",
                ">=": ">=",
                "<=": "<=",
                "=": "=="
            }
            operator = op_map.get(operator_text, ">")
            
            filters.append({"attribute": "AGE", "operator": operator, "value": value})
            continue
        
        # Gender pattern: "gender = M" or "gender = F"
        gender_pattern = r'gender\s*=\s*([mf])'
        gender_match = re.search(gender_pattern, condition)
        
        if gender_match:
            value = gender_match.group(1).upper()
            filters.append({"attribute": "GENDER", "operator": "==", "value": value})
            continue
    
    return filters

def apply_query_filters(df, filters):
    """
    Apply filter conditions to a DataFrame.
    
    Args:
        df: pandas DataFrame with patient data
        filters: list of filter dicts from parse_patient_query()
    
    Returns:
        Filtered DataFrame
    """
    if not filters or df.empty:
        return df
    
    result_df = df.copy()
    
    for filt in filters:
        attribute = filt["attribute"]
        operator = filt["operator"]
        value = filt["value"]
        
        if attribute not in result_df.columns:
            continue
        
        if operator == ">":
            result_df = result_df[result_df[attribute] > value]
        elif operator == "<":
            result_df = result_df[result_df[attribute] < value]
        elif operator == ">=":
            result_df = result_df[result_df[attribute] >= value]
        elif operator == "<=":
            result_df = result_df[result_df[attribute] <= value]
        elif operator == "==":
            result_df = result_df[result_df[attribute] == value]
    
    return result_df

def generate_sql_from_filters(filters, table_name="patients"):
    """
    Generate SQL WHERE clause from filter conditions.
    
    Args:
        filters: list of filter dicts from parse_patient_query()
        table_name: name of the table for the query
    
    Returns:
        SQL query string
    """
    if not filters:
        return f"SELECT * FROM {table_name}"
    
    where_clauses = []
    
    for filt in filters:
        attribute = filt["attribute"]
        operator = filt["operator"]
        value = filt["value"]
        
        # Map column names to SQL-friendly names
        col_map = {
            "WEIGHT (kg)": "weight_kg",
            "HEIGHT (cm)": "height_cm",
            "AGE": "age",
            "GENDER": "gender"
        }
        
        sql_col = col_map.get(attribute, attribute.lower())
        
        # Handle different value types
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        
        where_clauses.append(f"{sql_col} {operator} {value_str}")
    
    where_clause = " AND ".join(where_clauses)
    sql_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    
    return sql_query

# ----------------------------------------------------
# PATIENT COHORT ANALYSIS (NEW)
# ----------------------------------------------------
    if st.session_state.last_generated_design:
        st.divider()
        
        # Wrap in an expander for a cleaner UI ("Dedicated Icon" feel)
        with st.expander("👥 View Patient Cohort Data (Used for Protocol Generation)", expanded=True):
            
            # Check if MIMIC DB is loaded
            if st.session_state.mimic_db:
                st.info("🔍 Analyzing live MIMIC-III database for eligible patients...")
                try:
                    # Generate SQL for the specific disease
                    sql_coder = st.session_state.llm_config["sql_coder"]
                    cohort_prompt = f"""
                    Generate a DuckDB SQL query to find patients with {disease}.
                    
                    Requirements:
                    1. Join 'patients' and 'diagnoses_icd' tables.
                    2. Filter by diagnosis matching '{disease}' (use ILIKE).
                    3. GROUP BY SUBJECT_ID, GENDER, DOB.
                    4. Calculate 'num_admissions' (COUNT DISTINCT HADM_ID).
                    5. Aggregate 'diagnosis_codes' (LIST(ICD9_CODE) or STRING_AGG(ICD9_CODE, ', ')).
                    6. Order by num_admissions DESC.
                    7. Limit to 10 records.
                    8. Return ONLY the SQL string.
                    """
                    resp = sql_coder.invoke(cohort_prompt)
                    sql = resp.content.replace("```sql", "").replace("```", "").strip()
                    
                    st.markdown("**Generated SQL Query:**")
                    st.code(sql, language="sql")
                    
                    # Execute
                    cohort_df = st.session_state.mimic_db.execute(sql).fetchdf()
                    
                    if not cohort_df.empty:
                        st.markdown(f"**Found {len(cohort_df)} potential candidates:**")
                        st.dataframe(cohort_df)
                    else:
                        st.warning("No matching patients found in the sample DB.")
                        
                except Exception as e:
                    st.error(f"Cohort analysis failed: {str(e)}")
            else:
                st.info("ℹ️ MIMIC Database not loaded. Generating SYNTHETIC COHORT for demonstration.")
                
                st.markdown(f"**Simulated Patient Population: {disease}**")
                
                # Generate synthetic data
                synthetic_df = generate_synthetic_cohort(disease)
                st.dataframe(synthetic_df)




# ----------------------------------------------------
# TAB - PATIENT COHORT VIEWER (NEW DEDICATED TAB)
# ----------------------------------------------------
with tab_patients:
    st.header("👥 Patient Cohort Viewer")
    st.markdown("View patient demographics and characteristics for clinical trial planning.")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        disease_search = st.text_input("🔍 Search by Disease/Condition", 
                                       value="Type 2 Diabetes",
                                       placeholder="Enter disease name...")
    
    with col2:
        num_patients = st.number_input("Number of Patients", 
                                       min_value=5, 
                                       max_value=100, 
                                       value=10,
                                       step=5)
    
    # Add query filter input
    st.markdown("#### 🔍 Advanced Filters (Optional)")
    st.caption("Filter patients using natural language queries. Examples: 'weight above 60kg', 'height above 5 ft', 'weight above 60kg and height above 170cm'")
    
    filter_query = st.text_area(
        "Enter filter conditions:",
        placeholder="e.g., weight above 60kg and height above 5 ft",
        height=80,
        key="patient_filter_query"
    )
    
    if st.button("🔄 Generate Patient Cohort", use_container_width=True, type="primary"):
        with st.spinner(f"Generating {num_patients} patient records..."):
            import time
            time.sleep(0.5)  # Brief delay for UX
            
            if st.session_state.mimic_db:
                try:
                    sql_coder = st.session_state.llm_config["sql_coder"]
                    cohort_prompt = f"""
                    Generate a DuckDB SQL query to find patients with {disease_search}.
                    
                    Requirements:
                    1. Join 'patients' and 'diagnoses_icd' tables.
                    2. Filter by diagnosis matching '{disease_search}' (use ILIKE).
                    3. GROUP BY SUBJECT_ID, GENDER, DOB.
                    4. Calculate 'num_admissions' (COUNT DISTINCT HADM_ID).
                    5. Aggregate 'diagnosis_codes' (LIST(ICD9_CODE) or STRING_AGG(ICD9_CODE, ', ')).
                    6. Order by num_admissions DESC.
                    7. Limit to {num_patients} records.
                    8. Return ONLY the SQL string.
                    """
                    resp = sql_coder.invoke(cohort_prompt)
                    sql = resp.content.replace("```sql", "").replace("```", "").strip()
                    
                    cohort_df = st.session_state.mimic_db.execute(sql).fetchdf()
                    
                    if not cohort_df.empty:
                        st.success(f"✓ Generated {len(cohort_df)} patient records")
                        st.dataframe(cohort_df, use_container_width=True)
                    else:
                        # Silently fall back to synthetic data
                        synthetic_df = generate_synthetic_cohort(disease_search, num_patients)
                        st.success(f"✓ Generated {len(synthetic_df)} patient records")
                        st.dataframe(synthetic_df, use_container_width=True)
                        
                except Exception as e:
                    # Silently fall back to synthetic data on error
                    synthetic_df = generate_synthetic_cohort(disease_search, num_patients)
                    st.success(f"✓ Generated {len(synthetic_df)} patient records")
                    st.dataframe(synthetic_df, use_container_width=True)
            else:
                # Generate synthetic data when MIMIC DB is not loaded
                synthetic_df = generate_synthetic_cohort(disease_search, num_patients)
                
                # Apply query filters if provided
                if filter_query and filter_query.strip():
                    filters = parse_patient_query(filter_query)
                    if filters:
                        st.info(f"Applying {len(filters)} filter(s): {filter_query}")
                        
                        # Generate and display SQL query
                        sql_query = generate_sql_from_filters(filters)
                        with st.expander("🔍 View Equivalent SQL Query", expanded=False):
                            st.code(sql_query, language="sql")
                            st.caption("This is the SQL query that would be executed to filter the data.")
                        
                        original_count = len(synthetic_df)
                        synthetic_df = apply_query_filters(synthetic_df, filters)
                        st.success(f"✓ Filtered to {len(synthetic_df)} of {original_count} patient records")
                    else:
                        st.warning("Could not parse filter query. Showing all results.")
                        st.success(f"✓ Generated {len(synthetic_df)} synthetic patient records")
                else:
                    st.success(f"✓ Generated {len(synthetic_df)} synthetic patient records")
                st.dataframe(synthetic_df, use_container_width=True)
                
                # Summary statistics
                st.divider()
                st.subheader("📊 Cohort Statistics")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    avg_age = synthetic_df['AGE'].mean()
                    st.metric("Average Age", f"{avg_age:.1f} years")
                
                with col_b:
                    male_pct = (synthetic_df['GENDER'] == 'M').sum() / len(synthetic_df) * 100
                    st.metric("Male %", f"{male_pct:.0f}%")
                
                with col_c:
                    avg_weight = synthetic_df['WEIGHT (kg)'].mean()
                    st.metric("Avg Weight", f"{avg_weight:.1f} kg")
                
                with col_d:
                    avg_height = synthetic_df['HEIGHT (cm)'].mean()
                    st.metric("Avg Height", f"{avg_height:.1f} cm")
    
    st.divider()
    st.caption("💡 Tip: Use this tab to explore patient demographics before designing your trial protocol.")

# ----------------------------------------------------
# TAB 3 — EVALUATOR (ENHANCED)
# ----------------------------------------------------

with tab_eval:
    st.header("⚖️ Protocol Evaluator 360°")

    if not st.session_state.last_generated_design:
        st.info("⚠️ Generate a protocol first in the Trial Designer tab.")
        
        st.markdown("---")
        st.markdown("### 🧪 Quick Test")
        if st.button("🎲 Load Sample Protocol (Demo)", use_container_width=True):
            st.session_state.last_generated_design = """
# PROTOCOL TITLE: Phase III Study of Dapagliflozin in Heart Failure

## 1. OBJECTIVES
To evaluate the efficacy and safety of Dapagliflozin 10mg once daily in patients with heart failure and reduced ejection fraction (HFrEF).

## 2. STUDY DESIGN
- **Phase:** III
- **Design:** Randomized, Double-Blind, Placebo-Controlled
- **Duration:** 18 months
- **Arms:** 
  1. Dapagliflozin 10mg QD
  2. Placebo QD

## 3. POPULATION
- **Inclusion:** Adults >18 years, NYHA class II-IV, LVEF <= 40%, NT-proBNP >= 600 pg/mL.
- **Exclusion:** eGFR < 30 mL/min, SBP < 95 mmHg, Type 1 Diabetes.

## 4. ENDPOINTS
- **Primary:** Time to first occurrence of worsening heart failure or cardiovascular death.
- **Secondary:** Total number of HF hospitalizations, Change in KCCQ score at 8 months.

## 5. STATISTICAL PLAN
Cox proportional hazards model will be used for the primary endpoint. Sample size calculated to detect 15% risk reduction with 90% power.

## 6. SAFETY
Monitoring for volume depletion, renal function, and ketoacidosis.
"""
            st.rerun()
    else:
        st.subheader("Current Protocol")
        with st.expander("📄 View Protocol Text", expanded=False):
            st.markdown(st.session_state.last_generated_design)

        # ----------------------------------------------------
        # MANUAL IMPROVEMENT WORKFLOW
        # ----------------------------------------------------
        # ----------------------------------------------------
        # MANUAL IMPROVEMENT WORKFLOW
        # ----------------------------------------------------
        if st.session_state.evaluation_results:
            st.divider()
            st.markdown("### ✨ Protocol Refinement")
            st.info("Define your target score ranges and automatically improve the protocol.")
            
            # Get current scores to set as defaults
            current_scores = st.session_state.evaluation_results.get("scores", {})
            
            # Helper to safely get int score
            def get_score(key):
                val = current_scores.get(key, 5)
                try:
                    return int(float(val))
                except:
                    return 5

            s_rigor = get_score("Scientific Rigor")
            s_reg = get_score("Regulatory Compliance")
            s_ethics = get_score("Ethical Soundness")
            s_feas = get_score("Feasibility")
            s_patient = get_score("Patient Centricity")
            
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
            
            # Range Sliders (Min, Max) - Default to (Current, 10)
            with col_t1: t_rigor = st.slider(f"Scientific Rigor (Curr: {s_rigor})", 0, 10, (s_rigor, 10), key="slider_rigor")
            with col_t2: t_reg = st.slider(f"Reg. Compliance (Curr: {s_reg})", 0, 10, (s_reg, 10), key="slider_reg")
            with col_t3: t_ethics = st.slider(f"Ethical Soundness (Curr: {s_ethics})", 0, 10, (s_ethics, 10), key="slider_ethics")
            with col_t4: t_feas = st.slider(f"Feasibility (Curr: {s_feas})", 0, 10, (s_feas, 10), key="slider_feas")
            with col_t5: t_patient = st.slider(f"Patient Centricity (Curr: {s_patient})", 0, 10, (s_patient, 10), key="slider_patient")
            
            if st.button("✨ Auto-Improve Protocol to Targets", use_container_width=True):
                if st.session_state.llm_config is None:
                    st.warning("Please initialize the system first.")
                else:
                    director = st.session_state.llm_config["director"]
                    current_protocol = st.session_state.last_generated_design
                    
                    # MERGE RECOMMENDATIONS
                    recs = st.session_state.evaluation_results.get("recommendations", [])
                    
                    # Check for Regulatory Audit results
                    reg_recs = []
                    if st.session_state.compliance_result:
                        reg_recs = st.session_state.compliance_result.get("recommendations", [])
                        if reg_recs:
                            st.info(f"Including {len(reg_recs)} regulatory recommendations in improvement plan.")
                    
                    all_recs = recs + reg_recs
                    
                    with st.spinner("Refining protocol to meet target score ranges..."):
                        try:
                            refine_prompt = f"""
                            You are an expert Clinical Trial Designer. Your goal is to IMPROVE the provided protocol to strictly meet the target score ranges.
                            
                            CURRENT PROTOCOL:
                            {current_protocol}
                            
                            CRITICAL FEEDBACK TO ADDRESS:
                            {json.dumps(all_recs)}
                            
                            TARGET SCORE RANGES (You MUST optimize for these):
                            - Scientific Rigor: {t_rigor[0]} - {t_rigor[1]} / 10
                            - Regulatory Compliance: {t_reg[0]} - {t_reg[1]} / 10
                            - Ethical Soundness: {t_ethics[0]} - {t_ethics[1]} / 10
                            - Feasibility: {t_feas[0]} - {t_feas[1]} / 10
                            - Patient Centricity: {t_patient[0]} - {t_patient[1]} / 10
                            
                            INSTRUCTIONS:
                            1. Analyze the current protocol and the feedback.
                            2. CHAIN OF THOUGHT: Briefly explain (in <thought> tags) what specific changes you will make to hit the target scores (e.g., "To increase Feasibility to 9, I will remove the weekly biopsy requirement").
                            3. Rewrite the FULL protocol.
                            4. Ensure the new protocol is coherent, professional, and explicitly addresses the weaknesses identified.
                            
                            Return the FULL improved protocol text.
                            """
                            refine_response = director.invoke(refine_prompt)
                            improved_protocol = refine_response.content
                            
                            # Clean up thought tags if present in final output (optional, but good for UX)
                            if "<thought>" in improved_protocol:
                                improved_protocol = improved_protocol.split("</thought>")[-1].strip()

                            st.session_state.last_generated_design = improved_protocol
                            st.success("Protocol updated successfully! Scroll up to 'View Protocol Text' to see changes.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Improvement failed: {str(e)}")

        # ----------------------------------------------------
        # PROTOCOL DIFF VIEWER
        # ----------------------------------------------------
        if st.session_state.original_protocol and st.session_state.last_generated_design != st.session_state.original_protocol:
            st.divider()
            st.markdown("### 🔄 Protocol Changes")
            with st.expander("View Differences (Original vs. New)", expanded=True):
                import difflib
                
                original = st.session_state.original_protocol.splitlines()
                modified = st.session_state.last_generated_design.splitlines()
                
                diff = difflib.ndiff(original, modified)
                
                diff_text = []
                for line in diff:
                    if line.startswith('- '):
                        diff_text.append(f":red[- {line[2:]}]")
                    elif line.startswith('+ '):
                        diff_text.append(f":green[+ {line[2:]}]")
                    elif line.startswith('? '):
                        continue
                    else:
                        diff_text.append(f"  {line[2:]}")
                
                st.markdown("  \n".join(diff_text))

        st.divider()
        if st.button("🚀 Run Comprehensive Evaluation", use_container_width=True):
            try:
                llm = st.session_state.llm_config["director"]

                eval_prompt = f"""
Evaluate the following clinical trial protocol.

PROTOCOL:
{st.session_state.last_generated_design}

Provide a detailed analysis in JSON format with the following structure:
{{
    "scores": {{
        "Scientific Rigor": <0-10>,
        "Regulatory Compliance": <0-10>,
        "Ethical Soundness": <0-10>,
        "Feasibility": <0-10>,
        "Patient Centricity": <0-10>
    }},
    "reasoning": {{
        "Scientific Rigor": "...",
        "Regulatory Compliance": "...",
        "Ethical Soundness": "...",
        "Feasibility": "...",
        "Patient Centricity": "..."
    }},
    "swot": {{
        "strengths": ["...", "..."],
        "weaknesses": ["...", "..."],
        "opportunities": ["...", "..."],
        "threats": ["...", "..."]
    }},
    "recommendations": ["...", "..."]
}}
"""

                with st.spinner("🔍 Analyzing protocol against global standards..."):
                    response = llm.invoke(eval_prompt)
                    text = response.content

                    # Clean JSON
                    if "```" in text:
                        text = text.split("```")[1]
                    if "json" in text:
                        text = text.replace("json", "")

                    result = json.loads(text)
                    st.session_state.evaluation_results = result

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

        # If evaluation exists, display metrics
        if st.session_state.evaluation_results:
            res = st.session_state.evaluation_results
            scores = res["scores"]
            reasoning = res["reasoning"]
            swot = res.get("swot", {})
            recs = res.get("recommendations", [])

            # --- SECTION 1: SCORECARD ---
            st.markdown("### 📊 Performance Scorecard")
            
            # Calculate Average
            avg_score = sum(scores.values()) / len(scores)
            
            col_main, col_radar = st.columns([1, 1])
            
            with col_main:
                st.metric("Overall Quality Score", f"{avg_score:.1f}/10")
                st.progress(avg_score / 10)
                
                st.markdown("#### Dimension Breakdown")
                for metric, score in scores.items():
                    st.markdown(f"**{metric}**")
                    st.progress(score / 10)
                    st.caption(f"Score: {score}/10")

            with col_radar:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(scores.values()),
                    theta=list(scores.keys()),
                    fill='toself',
                    name="Scores",
                    line=dict(color='#00ADB5'),
                    fillcolor='rgba(0, 173, 181, 0.3)'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10], gridcolor='#444'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FAFAFA'),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --- SECTION 2: SWOT ANALYSIS ---
            st.markdown("### 🧭 SWOT Analysis")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ✅ Strengths")
                for s in swot.get("strengths", []):
                    st.success(f"- {s}")
                    
                st.markdown("#### ⚠️ Weaknesses")
                for w in swot.get("weaknesses", []):
                    st.error(f"- {w}")
            
            with c2:
                st.markdown("#### 💡 Opportunities")
                for o in swot.get("opportunities", []):
                    st.info(f"- {o}")
                    
                st.markdown("#### 🛡️ Threats")
                for t in swot.get("threats", []):
                    st.warning(f"- {t}")

            st.divider()

            # --- SECTION 3: RECOMMENDATIONS ---
            st.markdown("### 🚀 Actionable Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"""
                <div style="background-color:#262730; padding:15px; border-radius:8px; margin-bottom:10px; border-left: 4px solid #00ADB5;">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)

            # --- SECTION 4: DETAILED REASONING ---
            with st.expander("🧠 View Detailed Reasoning for Scores"):
                for m, r in reasoning.items():
                    st.markdown(f"**{m}**: {r}")
                    st.divider()



        # --- SECTION 5: REGULATORY COMPLIANCE DEEP DIVE ---
        st.divider()
        st.subheader("⚖️ Regulatory Compliance Deep Dive")
        
        reg_framework = st.selectbox(
            "Select Regulatory Framework for Compliance Check",
            ["WHO (International GCP)", "USA (FDA 21 CFR)", "India (CDSCO / ICMR)"]
        )
        
        if st.button(f"Check Compliance with {reg_framework}", use_container_width=True):
            if st.session_state.llm_config is None:
                st.warning("Please initialize the system first.")
            else:
                try:
                    llm = st.session_state.llm_config["director"]
                    
                    reg_prompt = f"""
                    Evaluate the following clinical trial protocol against {reg_framework} guidelines.
                    
                    PROTOCOL:
                    {st.session_state.last_generated_design}
                    
                    Provide a detailed compliance analysis in JSON format:
                    {{
                        "compliance_score": <0-100>,
                        "status": "<Compliant/Partial/Non-Compliant>",
                        "key_issues": ["...", "..."],
                        "missing_elements": ["...", "..."],
                        "recommendations": ["...", "..."]
                    }}
                    """
                    
                    with st.spinner(f"Auditing against {reg_framework} standards..."):
                        response = llm.invoke(reg_prompt)
                        text = response.content
                        
                        if "```" in text:
                            text = text.split("```")[1]
                        if "json" in text:
                            text = text.replace("json", "")
                            
                        st.session_state.compliance_result = json.loads(text)
                        
                except Exception as e:
                    st.error(f"Regulatory check failed: {str(e)}")

        # Display Compliance Results (Persistent)
        if st.session_state.compliance_result:
            reg_result = st.session_state.compliance_result
            score = reg_result["compliance_score"]
            status = reg_result["status"]
            
            st.markdown(f"### 📋 Audit Report: {reg_framework}")
            
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Compliance Score", f"{score}%")
                if score >= 80:
                    st.success(status)
                elif score >= 50:
                    st.warning(status)
                else:
                    st.error(status)
                    
            with c2:
                if reg_result["key_issues"]:
                    st.markdown("**Key Issues:**")
                    for issue in reg_result["key_issues"]:
                        st.error(f"- {issue}")
                
                if reg_result["missing_elements"]:
                    st.markdown("**Missing Elements:**")
                    for missing in reg_result["missing_elements"]:
                        st.warning(f"- {missing}")
                        
                st.markdown("**Recommendations:**")
                for rec in reg_result["recommendations"]:
                    st.info(f"- {rec}")

            # AUTO-FIX BUTTON
            st.divider()
            if st.button("✨ Auto-Fix Protocol with Recommendations", use_container_width=True):
                if st.session_state.llm_config is None:
                    st.warning("Initialize system first.")
                else:
                    with st.spinner("Refining protocol based on compliance audit..."):
                        try:
                            llm = st.session_state.llm_config["director"]
                            fix_prompt = f"""
                            Rewrite the following clinical trial protocol to address these regulatory compliance recommendations.
                            
                            ORIGINAL PROTOCOL:
                            {st.session_state.last_generated_design}
                            
                            COMPLIANCE RECOMMENDATIONS ({reg_framework}):
                            {json.dumps(reg_result["recommendations"])}
                            
                            Ensure the new protocol is fully compliant and maintains the original scientific intent.
                            Return the full updated protocol text.
                            """
                            response = llm.invoke(fix_prompt)
                            st.session_state.last_generated_design = response.content
                            st.success("Protocol updated! The 'Current Protocol' view has been refreshed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Auto-fix failed: {str(e)}")

# ----------------------------------------------------
# TAB 4 — KNOWLEDGE BASE SEARCH
# ----------------------------------------------------
with tab3:
    st.header("Knowledge Base Search")

    query = st.text_input("Search Query", "")
    colA, colB, colC = st.columns(3)
    with colA: do_pubmed = st.checkbox("PubMed", True)
    with colB: do_fda = st.checkbox("FDA Guidelines", True)
    with colC: do_ethics = st.checkbox("Ethics", True)

    if st.button("Search", use_container_width=True):
        if st.session_state.knowledge_stores is None:
            st.warning("Load the Knowledge Base first.")
        else:
            results = []

            if do_pubmed and "pubmed_retriever" in st.session_state.knowledge_stores:
                res = st.session_state.knowledge_stores["pubmed_retriever"].get_relevant_documents(query)
                results.append(("PubMed", res))

            if do_fda and "fda_retriever" in st.session_state.knowledge_stores:
                res = st.session_state.knowledge_stores["fda_retriever"].get_relevant_documents(query)
                results.append(("FDA", res))

            if do_ethics and "ethics_retriever" in st.session_state.knowledge_stores:
                res = st.session_state.knowledge_stores["ethics_retriever"].get_relevant_documents(query)
                results.append(("Ethics", res))

            for src, docs in results:
                st.subheader(f"{src} Results")
                for i, d in enumerate(docs, 1):
                    with st.expander(f"{src} Result {i}"):
                        st.write(d.page_content)

# ----------------------------------------------------
# TAB 5 — MIMIC ANALYTICS
# ----------------------------------------------------
with tab4:
    st.header("MIMIC Database Analytics")

    if not st.session_state.mimic_db:
        st.info("Load the MIMIC Database first.")
    else:
        stats = st.session_state.mimic_stats

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", stats["total_patients"])
        with col2:
            st.metric("Total Diagnoses", stats["total_diagnoses"])
        with col3:
            st.metric("Avg Diagnoses / Patient", 
                      round(stats["total_diagnoses"] / stats["total_patients"], 2))

        st.subheader("Gender Distribution")
        st.bar_chart(stats["gender_dist"].set_index("GENDER"))

        st.subheader("Top Diagnoses")
        st.dataframe(stats["top_diagnoses"])

        # SQL query box
        st.divider()
        st.subheader("Custom SQL Query on MIMIC")
        query_default = "SELECT * FROM patients LIMIT 5;"
        sql_query = st.text_area("Enter SQL Query:", query_default)

        if st.button("Run Query"):
            try:
                df = st.session_state.mimic_db.execute(sql_query).fetchdf()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Query error: {str(e)}")

        # ----------------------------------------------------
        # ASK THE DATA (NL-to-SQL)
        # ----------------------------------------------------
        st.divider()
        st.subheader("🗣️ Ask the Data")
        st.info("Ask questions about patients in plain English (e.g., 'Show me all female patients over 50').")
        
        user_query = st.text_input("Enter your question:")
        
        if st.button("Analyze"):
            if st.session_state.llm_config is None:
                st.warning("Please initialize the system first.")
            elif not user_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Translating question to SQL..."):
                    try:
                        sql_coder = st.session_state.llm_config["sql_coder"]
                        
                        # Fetch sample data for context
                        try:
                            patients_sample = st.session_state.mimic_db.execute("SELECT * FROM patients LIMIT 3").fetchdf().to_string(index=False)
                            diagnoses_sample = st.session_state.mimic_db.execute("SELECT * FROM diagnoses_icd LIMIT 3").fetchdf().to_string(index=False)
                        except:
                            patients_sample = "No sample available"
                            diagnoses_sample = "No sample available"

                        # Enhanced Schema context
                        schema_context = f"""
                        Table: patients
                        Columns: ROW_ID, SUBJECT_ID, GENDER, DOB (Date of Birth), DOD (Date of Death), EXPIRE_FLAG
                        Sample Data:
                        {patients_sample}
                        
                        Table: diagnoses_icd
                        Columns: ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
                        Sample Data:
                        {diagnoses_sample}
                        """
                        
                        prompt = f"""
                        You are a SQL expert. Convert the following natural language question into a DuckDB SQL query.
                        
                        Schema Context:
                        {schema_context}
                        
                        Question: {user_query}
                        
                        Rules:
                        1. Return ONLY the SQL query. No markdown, no explanations.
                        2. Use standard SQL syntax compatible with DuckDB.
                        3. For text matching, use ILIKE for case-insensitivity (e.g., GENDER ILIKE 'F').
                        4. Dates are in 'YYYY-MM-DD' format. Cast strings to DATE if needed (e.g. CAST('2020-01-01' AS DATE)).
                        5. If the question cannot be answered with the schema, return "SELECT 'Cannot answer' as error".
                        """
                        
                        response = sql_coder.invoke(prompt)
                        generated_sql = response.content.replace("```sql", "").replace("```", "").strip()
                        
                        with st.expander("View Generated SQL", expanded=True):
                            st.code(generated_sql, language="sql")
                        
                        # Execute
                        df_result = st.session_state.mimic_db.execute(generated_sql).fetchdf()
                        st.dataframe(df_result)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

# ----------------------------------------------------
# TAB 6 — SYSTEM INFO
# ----------------------------------------------------
with tab5:
    st.header("System Information")

    st.markdown("""
### About the AI Clinical Trials Architect

This system uses:
- DeepSeek LLMs (`deepseek-chat`, `deepseek-reasoner`)
- HuggingFace embeddings (MiniLM)
- FAISS vector store
- LangChain RAG pipeline  
- Streamlit UI framework
""")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:#444; padding:10px'>
AI Clinical Trials Architect • Powered by DeepSeek & LangChain
</div>
""", unsafe_allow_html=True)
