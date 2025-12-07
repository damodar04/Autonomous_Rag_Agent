"""
The AI Clinical Trials Architect: A Self-Evolving Agent Guild for Multi-Objective RAG Optimization

This module implements a hierarchical agent-of-agents system for automating clinical trial design,
specifically drafting Patient Inclusion/Exclusion Criteria.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel as LangChainBaseModel

# Try importing langgraph, handle gracefully if not available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except (TypeError, ImportError) as e:
    print(f"Warning: LangGraph import issue: {e}")
    print("LangGraph functionality will be limited. Consider updating langgraph package.")
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

# Bio and database imports
from Bio import Entrez, Medline
import duckdb
import requests
from pypdf import PdfReader
import io


# ================================
# Environment Configuration
# ================================

def setup_environment():
    """Load environment variables and configure the system."""
    load_dotenv()
    
    if "DEEPSEEK_API_KEY" not in os.environ or "ENTREZ_EMAIL" not in os.environ:
        print("Required environment variables not set. Please set them in your .env file or environment.")
        print("Required: DEEPSEEK_API_KEY, ENTREZ_EMAIL")
        return False
    else:
        print("Environment variables loaded successfully.")
    
    os.environ["LANGCHAIN_PROJECT"] = "AI_Clinical_Trials_Architect"
    if "LANGCHAIN_API_KEY" in os.environ:
        print(f"LangSmith tracing is configured for project '{os.environ['LANGCHAIN_PROJECT']}'.")
    else:
        print("LangSmith tracing is disabled (no API key found).")
    
    return True


# ================================
# Model Configuration
# ================================

def configure_models():
    """Configure DeepSeek API models and embeddings."""
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    # Configure chat LLM clients (DeepSeek)
    llm_config = {
        "planner": ChatOpenAI(
            model="deepseek-chat",
            temperature=0.0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model_kwargs={"response_format": {"type": "json_object"}}
        ),
        "drafter": ChatOpenAI(
            model="deepseek-chat",
            temperature=0.2,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        ),
        "sql_coder": ChatOpenAI(
            model="deepseek-chat",
            temperature=0.0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        ),
        "director": ChatOpenAI(
            model="deepseek-reasoner",
            temperature=0.0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model_kwargs={"response_format": {"type": "json_object"}}
        ),
    }

    # Try to use HuggingFace / sentence-transformers for embeddings. If that fails
    # (missing packages, TensorFlow/PyTorch DLL issues), fall back to a lightweight
    # local TF-IDF based embedding class so the app can initialize and the KB
    # can still be used for basic similarity search.
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Using HuggingFaceEmbeddings (sentence-transformers).")
    except Exception as e:
        print("Warning: HuggingFace / sentence-transformers unavailable:", e)
        print("Falling back to LocalTfidfEmbeddings (no torch/tf required).")

        # Lightweight TF-IDF embeddings wrapper
        class LocalTfidfEmbeddings:
            def __init__(self):
                self.vectorizer = None

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                except Exception:
                    raise ImportError("sklearn is required for LocalTfidfEmbeddings")
                self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=768)
                X = self.vectorizer.fit_transform(texts)
                return X.toarray().tolist()

            def embed_query(self, text: str) -> List[float]:
                if self.vectorizer is None:
                    # If no docs were embedded before, fit on the query itself
                    return self.embed_documents([text])[0]
                X = self.vectorizer.transform([text])
                return X.toarray()[0].tolist()

        embedding_model = LocalTfidfEmbeddings()

    llm_config["embedding_model"] = embedding_model
    
    print("DeepSeek LLM clients configured:")
    print(f"Planner (deepseek-chat): {llm_config['planner']}")
    print(f"Drafter (deepseek-chat): {llm_config['drafter']}")
    print(f"SQL Coder (deepseek-chat): {llm_config['sql_coder']}")
    print(f"Director (deepseek-reasoner): {llm_config['director']}")
    print(f"Embedding Model (all-MiniLM-L6-v2): {llm_config['embedding_model']}")
    
    return llm_config


# ================================
# Data Preparation
# ================================

def create_data_directories():
    """Create necessary data directories."""
    data_paths = {
        "base": "./data",
        "pubmed": "./data/pubmed_articles",
        "fda": "./data/fda_guidelines",
        "ethics": "./data/ethical_guidelines",
        "mimic": "./data/mimic_db"
    }
    
    for path in data_paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
    
    return data_paths


def download_pubmed_articles(query, max_articles=20, data_paths=None):
    """Fetches abstracts from PubMed and saves them as text files."""
    Entrez.email = os.environ.get("ENTREZ_EMAIL")
    print(f"Fetching PubMed articles for query: {query}")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles, sort="relevance")
    record = Entrez.read(handle)
    id_list = record["IdList"]
    print(f"Found {len(id_list)} article IDs.")
    
    print("Downloading articles...")
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    
    count = 0
    for i, record in enumerate(records):
        pmid = record.get("PMID", "")
        title = record.get("TI", "No Title")
        abstract = record.get("AB", "No Abstract")
        if pmid:
            filepath = os.path.join(data_paths["pubmed"], f"{pmid}.txt")
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(f"Title: {title}\n\nAbstract: {abstract}")
            print(f"[{i+1}/{len(id_list)}] Fetching PMID: {pmid}... Saved to {filepath}")
            count += 1
    return count


def download_and_extract_text_from_pdf(url, output_path):
    """Download FDA guidelines PDF and extract text."""
    print(f"Downloading FDA Guideline: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded and saved to {output_path}")
        
        reader = PdfReader(io.BytesIO(response.content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        txt_output_path = os.path.splitext(output_path)[0] + '.txt'
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def create_ethics_document(data_paths):
    """Create a sample Clinical Trial Ethics document."""
    ethics_content = """
Title: Summary of the Belmont Report Principles for Clinical Research

1. Respect for Persons: This principle requires that individuals be treated as autonomous agents and that persons with diminished autonomy are entitled to protection. This translates to robust informed consent processes. Inclusion/exclusion criteria must not unduly target or coerce vulnerable populations, such as economically disadvantaged individuals, prisoners, or those with severe cognitive impairments, unless the research is directly intended to benefit that population.

2. Beneficence: This principle involves two complementary rules: (1) do not harm and (2) maximize possible benefits and minimize possible harms. The criteria must be designed to select a population that is most likely to benefit and least likely to be harmed by the intervention. The risks to subjects must be reasonable in relation to anticipated benefits.

3. Justice: This principle concerns the fairness of distribution of the burdens and benefits of research. The selection of research subjects must be equitable. Criteria should not be designed to exclude certain groups without a sound scientific or safety-related justification. For example, excluding participants based on race, gender, or socioeconomic status is unjust unless there is a clear rationale related to the drug's mechanism or risk profile.
"""
    
    ethics_path = os.path.join(data_paths["ethics"], "belmont_summary.txt")
    with open(ethics_path, "w", encoding='utf-8') as f:
        f.write(ethics_content)
    
    print(f"Created ethics guideline file: {ethics_path}")


def load_real_mimic_data(data_paths):
    """Loads real MIMIC-III CSVs into a DuckDB database."""
    print("Attempting to load real MIMIC-III data from local CSVs...")
    db_path = os.path.join(data_paths["mimic"], "mimic3_real.db")
    csv_dir = os.path.join(data_paths["mimic"], "mimiciii_csvs")
    
    required_files = {
        "patients": os.path.join(csv_dir, "PATIENTS.csv.gz"),
        "diagnoses": os.path.join(csv_dir, "DIAGNOSES_ICD.csv.gz"),
        "labevents": os.path.join(csv_dir, "LABEVENTS.csv.gz"),
    }
    
    missing_files = [path for path in required_files.values() if not os.path.exists(path)]
    if missing_files:
        print("ERROR: The following MIMIC-III files were not found:")
        for f in missing_files:
            print(f"- {f}")
        print("\nPlease download them as instructed and place them in the correct directory.")
        return None
    
    print("Required files found. Proceeding with database creation.")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    
    print(f"Loading {required_files['patients']} into DuckDB...")
    con.execute(f"CREATE TABLE patients AS SELECT SUBJECT_ID, GENDER, DOB, DOD FROM read_csv_auto('{required_files['patients']}')")
    
    print(f"Loading {required_files['diagnoses']} into DuckDB...")
    con.execute(f"CREATE TABLE diagnoses_icd AS SELECT SUBJECT_ID, ICD9_CODE FROM read_csv_auto('{required_files['diagnoses']}')")
    
    print(f"Loading and processing {required_files['labevents']} (this may take several minutes)...")
    con.execute(f"""CREATE TABLE labevents_staging AS 
                   SELECT SUBJECT_ID, ITEMID, VALUENUM 
                   FROM read_csv_auto('{required_files['labevents']}', all_varchar=True) 
                   WHERE ITEMID IN ('50912', '50852') AND VALUENUM IS NOT NULL AND VALUENUM ~ '^[0-9]+(\\.[0-9]+)?$'
                """)
    con.execute("CREATE TABLE labevents AS SELECT SUBJECT_ID, CAST(ITEMID AS INTEGER) AS ITEMID, CAST(VALUENUM AS DOUBLE) AS VALUENUM FROM labevents_staging")
    con.execute("DROP TABLE labevents_staging")

    con.close()
    return db_path


def create_vector_store(folder_path: str, embedding_model, store_name: str):
    """Loads documents from a folder, splits them, and creates a FAISS vector store."""
    print(f"--- Creating {store_name} Vector Store ---")
    loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {folder_path}")
        return None, 0, 0
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} documents, split into {len(texts)} chunks.")
    print("Generating embeddings and indexing into FAISS... (This may take a moment)")
    db = FAISS.from_documents(texts, embedding_model)
    print(f"{store_name} Vector Store created successfully.")
    return db, len(documents), len(texts)


def create_retrievers(embedding_model, data_paths, db_path):
    """Create all retrievers for the knowledge base."""
    pubmed_db, _, _ = create_vector_store(data_paths["pubmed"], embedding_model, "PubMed")
    fda_db, _, _ = create_vector_store(data_paths["fda"], embedding_model, "FDA")
    ethics_db, _, _ = create_vector_store(data_paths["ethics"], embedding_model, "Ethics")
    
    result = {}
    if pubmed_db:
        result["pubmed_retriever"] = pubmed_db.as_retriever(search_kwargs={"k": 3})
    if fda_db:
        result["fda_retriever"] = fda_db.as_retriever(search_kwargs={"k": 3})
    if ethics_db:
        result["ethics_retriever"] = ethics_db.as_retriever(search_kwargs={"k": 2})
    result["mimic_db_path"] = db_path
    
    return result


# ================================
# Guild SOP & State Definitions
# ================================

class GuildSOP(BaseModel):
    """Standard Operating Procedures for the Trial Design Guild."""
    planner_prompt: str = Field(description="The system prompt for the Planner Agent.")
    researcher_retriever_k: int = Field(description="Number of documents for the Medical Researcher to retrieve.", default=3)
    synthesizer_prompt: str = Field(description="The system prompt for the Criteria Synthesizer Agent.")
    synthesizer_model: Literal["deepseek-chat", "deepseek-reasoner"] = Field(description="The DeepSeek model to use for the Synthesizer.", default="deepseek-chat")
    use_sql_analyst: bool = Field(description="Whether to use the Patient Cohort Analyst agent.", default=True)
    use_ethics_specialist: bool = Field(description="Whether to use the Ethics Specialist agent.", default=True)


class AgentOutput(LangChainBaseModel):
    """A structured output for each agent's findings."""
    agent_name: str
    findings: Any


class GuildState(TypedDict):
    """The state of the Trial Design Guild's workflow."""
    initial_request: str
    plan: Optional[Dict[str, Any]]
    agent_outputs: List[AgentOutput]
    final_criteria: Optional[str]
    sop: GuildSOP


# ================================
# Agent Functions
# ================================

def planner_agent(state: GuildState, llm_config) -> GuildState:
    """Receives the initial request and creates a plan."""
    print("--- EXECUTING PLANNER AGENT ---")
    sop = state['sop']
    planner_llm = llm_config['planner']
    
    prompt = f"{sop.planner_prompt}\n\nTrial Concept: '{state['initial_request']}'"
    print(f"Planner Prompt:\n{prompt}")
    
    response = planner_llm.invoke(prompt)
    
    result = json.loads(response.content)
    print(f"Generated Plan:\n{json.dumps(result, indent=2)}")
    
    return {**state, "plan": result}


def retrieval_agent(task_description: str, state: GuildState, retriever_name: str, agent_name: str, knowledge_stores) -> AgentOutput:
    """Generic agent to perform retrieval from a specified vector store."""
    print(f"--- EXECUTING {agent_name.upper()} ---")
    print(f"Task: {task_description}")
    retriever = knowledge_stores[retriever_name]
    
    if agent_name == "Medical Researcher":
        retriever.search_kwargs['k'] = state['sop'].researcher_retriever_k
        print(f"Using k={state['sop'].researcher_retriever_k} for retrieval.")

    retrieved_docs = retriever.invoke(task_description)
    
    findings = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in retrieved_docs])
    print(f"Retrieved {len(retrieved_docs)} documents.")
    print(f"Sample Finding:\n{findings[:500]}...")
    return AgentOutput(agent_name=agent_name, findings=findings)


def patient_cohort_analyst(task_description: str, state: GuildState, llm_config, knowledge_stores) -> AgentOutput:
    """Estimates cohort size by generating and executing a SQL query against the MIMIC database."""
    print("--- EXECUTING PATIENT COHORT ANALYST ---")
    if not state['sop'].use_sql_analyst:
        return AgentOutput(agent_name="Patient Cohort Analyst", findings="Analysis skipped as per SOP.")
    
    con = duckdb.connect(knowledge_stores['mimic_db_path'])
    schema_query = """
    SELECT table_name, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'main' ORDER BY table_name, column_name;
    """
    schema = con.execute(schema_query).df()
    con.close()
    
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert SQL writer specializing in DuckDB... The database contains patient data with the following schema:\n{schema.to_string()}\n\nIMPORTANT: All column names in your query MUST be uppercase (e.g., SELECT SUBJECT_ID, ICD9_CODE...).\n\nKey Mappings:\n- T2DM (Type 2 Diabetes) corresponds to ICD9_CODE '25000'.\n- Moderate renal impairment can be estimated by a creatinine lab value (ITEMID 50912) where VALUENUM is between 1.5 and 3.0.\n- Uncontrolled T2D can be estimated by an HbA1c lab value (ITEMID 50852) where VALUENUM is greater than 8.0."),
        ("human", "Please write a SQL query to count the number of unique patients who meet the following criteria: {task}")
    ])
    
    sql_chain = sql_generation_prompt | llm_config['sql_coder'] | StrOutputParser()
    
    print(f"Generating SQL for task: {task_description}")
    sql_query = sql_chain.invoke({"task": task_description})
    sql_query = sql_query.strip().replace("```sql", "").replace("```", "")
    print(f"Generated SQL Query:\n{sql_query}")

    try:
        con = duckdb.connect(knowledge_stores['mimic_db_path'])
        result = con.execute(sql_query).fetchone()
        patient_count = result[0] if result else 0
        con.close()
        
        findings = f"Generated SQL Query:\n{sql_query}\n\nEstimated eligible patient count from the synthetic database: {patient_count}."
        print(f"Query executed successfully. Estimated patient count: {patient_count}")
    except Exception as e:
        findings = f"Error executing SQL query: {e}. Defaulting to a count of 0."
        print(f"Error during query execution: {e}")

    return AgentOutput(agent_name="Patient Cohort Analyst", findings=findings)


def criteria_synthesizer(state: GuildState, llm_config) -> GuildState:
    """Synthesizes all findings into the final criteria document."""
    print("--- EXECUTING CRITERIA SYNTHESIZER ---")
    sop = state['sop']
    
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    
    drafter_llm = ChatOpenAI(
        model=sop.synthesizer_model,
        temperature=0.2,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )

    context = "\n\n---\n\n".join([f"**{out.agent_name} Findings:**\n{out.findings}" for out in state['agent_outputs']])
    
    prompt = f"{sop.synthesizer_prompt}\n\n**Context from Specialist Teams:**\n{context}"
    print(f"Synthesizer is using model '{sop.synthesizer_model}'.")

    response = drafter_llm.invoke(prompt)
    print("Final criteria generated.")
    
    return {**state, "final_criteria": response.content}


def specialist_execution_node(state: GuildState, llm_config, knowledge_stores) -> GuildState:
    """Executes all specialist tasks from the plan."""
    plan_tasks = state['plan']['plan']
    outputs = []
    
    for task in plan_tasks:
        agent_name = task['agent']
        task_desc = task['task_description']
        
        if "Regulatory" in agent_name:
            output = retrieval_agent(task_desc, state, "fda_retriever", "Regulatory Specialist", knowledge_stores)
        elif "Medical" in agent_name:
            output = retrieval_agent(task_desc, state, "pubmed_retriever", "Medical Researcher", knowledge_stores)
        elif "Ethics" in agent_name and state['sop'].use_ethics_specialist:
            output = retrieval_agent(task_desc, state, "ethics_retriever", "Ethics Specialist", knowledge_stores)
        elif "Cohort" in agent_name:
            output = patient_cohort_analyst(task_desc, state, llm_config, knowledge_stores)
        else:
            continue
        
        outputs.append(output)

    return {**state, "agent_outputs": outputs}


def build_guild_graph(llm_config, knowledge_stores):
    """Build and compile the Guild LangGraph."""
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph is not available. Cannot build guild graph.")
        return None
    
    workflow = StateGraph(GuildState)

    workflow.add_node("planner", lambda state: planner_agent(state, llm_config))
    workflow.add_node("execute_specialists", lambda state: specialist_execution_node(state, llm_config, knowledge_stores))
    workflow.add_node("synthesizer", lambda state: criteria_synthesizer(state, llm_config))

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "execute_specialists")
    workflow.add_edge("execute_specialists", "synthesizer")
    workflow.add_edge("synthesizer", END)

    guild_graph = workflow.compile()
    print("Graph compiled successfully.")
    
    return guild_graph


# ================================
# Evaluation Functions
# ================================

class GradedScore(BaseModel):
    score: float = Field(description="A score from 0.0 to 1.0")
    reasoning: str = Field(description="A brief justification for the score.")


def scientific_rigor_evaluator(generated_criteria: str, pubmed_context: str, llm_config) -> GradedScore:
    evaluator_llm = llm_config['director']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert clinical scientist. Evaluate a set of clinical trial criteria based on the provided scientific literature. A score of 1.0 means the criteria are perfectly aligned with and justified by the literature. A score of 0.0 means they contradict or ignore the literature. Respond in JSON format with keys: score (float 0.0-1.0), reasoning (string)."),
        ("human", "Evaluate the following criteria:\n\n**Generated Criteria:**\n{criteria}\n\n**Supporting Scientific Context:**\n{context}")
    ])
    chain = prompt | evaluator_llm
    response = chain.invoke({"criteria": generated_criteria, "context": pubmed_context})
    return GradedScore(**json.loads(response.content))


def regulatory_compliance_evaluator(generated_criteria: str, fda_context: str, llm_config) -> GradedScore:
    evaluator_llm = llm_config['director']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert regulatory affairs specialist. Evaluate if a set of clinical trial criteria adheres to the provided FDA guidelines. A score of 1.0 means full compliance. Respond in JSON format with keys: score (float 0.0-1.0), reasoning (string)."),
        ("human", "Evaluate the following criteria:\n\n**Generated Criteria:**\n{criteria}\n\n**Applicable FDA Guidelines:**\n{context}")
    ])
    chain = prompt | evaluator_llm
    response = chain.invoke({"criteria": generated_criteria, "context": fda_context})
    return GradedScore(**json.loads(response.content))


def ethical_soundness_evaluator(generated_criteria: str, ethics_context: str, llm_config) -> GradedScore:
    evaluator_llm = llm_config['director']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert on clinical trial ethics. Evaluate if a set of criteria adheres to the ethical principles provided (summarizing the Belmont Report). A score of 1.0 means the criteria show strong respect for persons, beneficence, and justice. Respond in JSON format with keys: score (float 0.0-1.0), reasoning (string)."),
        ("human", "Evaluate the following criteria:\n\n**Generated Criteria:**\n{criteria}\n\n**Ethical Principles:**\n{context}")
    ])
    chain = prompt | evaluator_llm
    response = chain.invoke({"criteria": generated_criteria, "context": ethics_context})
    return GradedScore(**json.loads(response.content))


def feasibility_evaluator(cohort_analyst_output: AgentOutput) -> GradedScore:
    findings_text = cohort_analyst_output.findings
    try:
        count_str = findings_text.split("database: ")[1].replace('.', '')
        patient_count = int(count_str)
    except (IndexError, ValueError):
        return GradedScore(score=0.0, reasoning="Could not parse patient count from analyst output.")
    
    IDEAL_COUNT = 150.0
    score = min(1.0, patient_count / IDEAL_COUNT)
    reasoning = f"Estimated {patient_count} eligible patients. Score is normalized against an ideal target of {int(IDEAL_COUNT)}."
    return GradedScore(score=score, reasoning=reasoning)


def simplicity_evaluator(generated_criteria: str) -> GradedScore:
    EXPENSIVE_TESTS = ["mri", "genetic sequencing", "pet scan", "biopsy", "echocardiogram", "endoscopy"]
    test_count = sum(1 for test in EXPENSIVE_TESTS if test in generated_criteria.lower())
    score = max(0.0, 1.0 - (test_count * 0.5))
    reasoning = f"Found {test_count} expensive/complex screening procedures mentioned."
    return GradedScore(score=score, reasoning=reasoning)


class EvaluationResult(BaseModel):
    rigor: GradedScore
    compliance: GradedScore
    ethics: GradedScore
    feasibility: GradedScore
    simplicity: GradedScore
    

def run_full_evaluation(guild_final_state: GuildState, llm_config) -> EvaluationResult:
    print("--- RUNNING FULL EVALUATION GAUNTLET ---")
    final_criteria = guild_final_state['final_criteria']
    agent_outputs = guild_final_state['agent_outputs']
    
    pubmed_context = next((o.findings for o in agent_outputs if o.agent_name == "Medical Researcher"), "")
    fda_context = next((o.findings for o in agent_outputs if o.agent_name == "Regulatory Specialist"), "")
    ethics_context = next((o.findings for o in agent_outputs if o.agent_name == "Ethics Specialist"), "")
    analyst_output = next((o for o in agent_outputs if o.agent_name == "Patient Cohort Analyst"), None)
    
    print("Evaluating: Scientific Rigor...")
    rigor = scientific_rigor_evaluator(final_criteria, pubmed_context, llm_config)
    print("Evaluating: Regulatory Compliance...")
    compliance = regulatory_compliance_evaluator(final_criteria, fda_context, llm_config)
    print("Evaluating: Ethical Soundness...")
    ethics = ethical_soundness_evaluator(final_criteria, ethics_context, llm_config)
    print("Evaluating: Recruitment Feasibility...")
    feasibility = feasibility_evaluator(analyst_output) if analyst_output else GradedScore(score=0, reasoning="Analyst did not run.")
    print("Evaluating: Operational Simplicity...")
    simplicity = simplicity_evaluator(final_criteria)
    
    print("--- EVALUATION GAUNTLET COMPLETE ---")
    return EvaluationResult(rigor=rigor, compliance=compliance, ethics=ethics, feasibility=feasibility, simplicity=simplicity)


# ================================
# Evolution Engine
# ================================

class SOPGenePool:
    """A class to store and manage a collection of GuildSOPs and their evaluations."""
    def __init__(self):
        self.pool: List[Dict[str, Any]] = []
        self.version_counter = 0

    def add(self, sop: GuildSOP, eval_result: EvaluationResult, parent_version: Optional[int] = None):
        self.version_counter += 1
        entry = {
            "version": self.version_counter,
            "sop": sop,
            "evaluation": eval_result,
            "parent": parent_version
        }
        self.pool.append(entry)
        print(f"Added SOP v{self.version_counter} to the gene pool.")
        
    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        return self.pool[-1] if self.pool else None


class Diagnosis(BaseModel):
    primary_weakness: Literal['rigor', 'compliance', 'ethics', 'feasibility', 'simplicity']
    root_cause_analysis: str = Field(description="A detailed analysis of why the weakness occurred, referencing specific scores.")
    recommendation: str = Field(description="A high-level recommendation for how to modify the SOP to address the weakness.")


def performance_diagnostician(eval_result: EvaluationResult, llm_config) -> Diagnosis:
    """Analyzes the 5D evaluation vector and diagnoses the primary weakness."""
    print("--- EXECUTING PERFORMANCE DIAGNOSTICIAN ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class management consultant specializing in process optimization. Your task is to analyze a performance scorecard and identify the single biggest weakness. Then, provide a root cause analysis and a strategic recommendation. Respond in JSON format with keys: primary_weakness, root_cause_analysis, recommendation."),
        ("human", "Please analyze the following performance evaluation report:\n\n{report}")
    ])
    
    diagnostician_llm = llm_config['director']
    chain = prompt | diagnostician_llm
    response = chain.invoke({"report": eval_result.json()})
    
    result_dict = json.loads(response.content)
    return Diagnosis(**result_dict)


class EvolvedSOPs(BaseModel):
    """A container for a list of new, evolved GuildSOPs."""
    mutations: List[GuildSOP]


def sop_architect(diagnosis: Diagnosis, current_sop: GuildSOP, llm_config) -> EvolvedSOPs:
    """Takes a diagnosis and the current SOP, and generates new, mutated SOPs."""
    print("--- EXECUTING SOP ARCHITECT ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an AI process architect. Your job is to modify a process configuration (an SOP) to fix a diagnosed problem. The SOP is a JSON object with this schema: {GuildSOP.schema_json()}. You must return a JSON object with a 'mutations' key containing a list of 2-3 new, valid SOP JSON objects. Propose diverse and creative mutations. For example, you can change prompts, toggle agents, change retrieval parameters, or even change the model used for a task. Only modify fields relevant to the diagnosis."),
        ("human", "Here is the current SOP:\n{current_sop}\n\nHere is the performance diagnosis:\n{diagnosis}\n\nBased on the diagnosis, please generate 2-3 new, improved SOPs.")
    ])
    
    architect_llm = llm_config['director']
    chain = prompt | architect_llm
    response = chain.invoke({"current_sop": current_sop.json(), "diagnosis": diagnosis.json()})
    
    result_dict = json.loads(response.content)
    mutations = [GuildSOP(**mutation) for mutation in result_dict['mutations']]
    return EvolvedSOPs(mutations=mutations)


def run_evolution_cycle(gene_pool: SOPGenePool, trial_request: str, llm_config, knowledge_stores, guild_graph):
    """Runs one full cycle of diagnosis, mutation, and evaluation."""
    print("\n" + "="*25 + " STARTING NEW EVOLUTION CYCLE " + "="*25)
    
    current_best_entry = gene_pool.get_latest_entry()
    parent_sop = current_best_entry['sop']
    parent_eval = current_best_entry['evaluation']
    parent_version = current_best_entry['version']
    print(f"Improving upon SOP v{parent_version}...")
    
    diagnosis = performance_diagnostician(parent_eval, llm_config)
    print(f"Diagnosis complete. Primary Weakness: '{diagnosis.primary_weakness}'. Recommendation: {diagnosis.recommendation}")

    new_sop_candidates = sop_architect(diagnosis, parent_sop, llm_config)
    print(f"Generated {len(new_sop_candidates.mutations)} new SOP candidates.")

    for i, candidate_sop in enumerate(new_sop_candidates.mutations):
        print(f"\n--- Testing SOP candidate {i+1}/{len(new_sop_candidates.mutations)} ---")
        guild_input = {"initial_request": trial_request, "sop": candidate_sop}
        final_state = guild_graph.invoke(guild_input)
        
        eval_result = run_full_evaluation(final_state, llm_config)
        gene_pool.add(sop=candidate_sop, eval_result=eval_result, parent_version=parent_version)

    print("\n" + "="*25 + " EVOLUTION CYCLE COMPLETE " + "="*26)


# ================================
# Pareto Front Analysis
# ================================

def identify_pareto_front(gene_pool: SOPGenePool) -> List[Dict[str, Any]]:
    """Identifies the non-dominated solutions in the gene pool."""
    pareto_front = []
    pool_entries = gene_pool.pool
    
    for i, candidate in enumerate(pool_entries):
        is_dominated = False
        cand_scores = np.array([s['score'] for s in candidate['evaluation'].dict().values()])
        
        for j, other in enumerate(pool_entries):
            if i == j: continue
            other_scores = np.array([s['score'] for s in other['evaluation'].dict().values()])
            
            if np.all(other_scores >= cand_scores) and np.any(other_scores > cand_scores):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(candidate)
            
    return pareto_front


def visualize_frontier(pareto_sops):
    """Creates a 2D scatter plot and a parallel coordinates plot for the Pareto front."""
    if not pareto_sops:
        print("No SOPs on the Pareto front to visualize.")
        return

    labels = [f"v{s['version']}" for s in pareto_sops]
    rigor_scores = [s['evaluation'].rigor.score for s in pareto_sops]
    feasibility_scores = [s['evaluation'].feasibility.score for s in pareto_sops]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(rigor_scores, feasibility_scores, s=150, alpha=0.7)
    for i, txt in enumerate(labels):
        ax1.annotate(txt, (rigor_scores[i], feasibility_scores[i]), xytext=(10,-10), textcoords='offset points', fontsize=12)
    ax1.set_title('Pareto Frontier: Rigor vs. Feasibility', fontsize=14)
    ax1.set_xlabel('Scientific Rigor Score', fontsize=12)
    ax1.set_ylabel('Recruitment Feasibility Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(min(rigor_scores)-0.05, max(rigor_scores)+0.05)
    ax1.set_ylim(min(feasibility_scores)-0.1, max(feasibility_scores)+0.1)

    data = []
    for s in pareto_sops:
        eval_dict = s['evaluation'].dict()
        scores = {k: v['score'] for k, v in eval_dict.items()}
        scores['SOP Version'] = f"v{s['version']}"
        data.append(scores)
    
    df = pd.DataFrame(data)
    pd.plotting.parallel_coordinates(df, 'SOP Version', colormap=plt.get_cmap("viridis"), ax=ax2, axvlines_kwargs={"linewidth": 1, "color": "grey"})
    ax2.set_title('5D Performance Trade-offs on Pareto Front', fontsize=14)
    ax2.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Normalized Score', fontsize=12)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# ================================
# Main Execution Function
# ================================

def main():
    """Main execution function."""
    print("="*70)
    print("AI CLINICAL TRIALS ARCHITECT - INITIALIZATION")
    print("="*70)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Configure models
    llm_config = configure_models()
    
    # Create data directories
    data_paths = create_data_directories()
    
    # Create baseline SOP
    baseline_sop = GuildSOP(
        planner_prompt="""You are a master planner for clinical trial design. Your task is to receive a high-level trial concept and break it down into a structured plan with specific sub-tasks for a team of specialists: a Regulatory Specialist, a Medical Researcher, an Ethics Specialist, and a Patient Cohort Analyst. Output a JSON object with a single key 'plan' containing a list of tasks. Each task must have 'agent', 'task_description', and 'dependencies' keys.""",
        synthesizer_prompt="""You are an expert medical writer. Your task is to synthesize the structured findings from all specialist teams into a formal 'Inclusion and Exclusion Criteria' document. Be concise, precise, and adhere strictly to the information provided. Structure your output into two sections: 'Inclusion Criteria' and 'Exclusion Criteria'.""",
        researcher_retriever_k=3,
        synthesizer_model="deepseek-chat",
        use_sql_analyst=True,
        use_ethics_specialist=True
    )
    
    print("\nBaseline GuildSOP (v1.0):")
    print(json.dumps(baseline_sop.dict(), indent=4))
    
    # Note: Data loading is optional and can be skipped if data already exists
    print("\n" + "="*70)
    print("SYSTEM READY - Use build_guild_graph() to create the workflow")
    print("="*70)
    
    return llm_config, data_paths, baseline_sop


if __name__ == "__main__":
    # Initialize the system
    llm_config, data_paths, baseline_sop = main()
    
    # Example usage (commented out - uncomment to run):
    """
    # Load or create knowledge stores
    db_path = load_real_mimic_data(data_paths)
    knowledge_stores = create_retrievers(llm_config["embedding_model"], data_paths, db_path)
    
    # Build the guild graph
    guild_graph = build_guild_graph(llm_config, knowledge_stores)
    
    # Run a test
    test_request = "Draft inclusion/exclusion criteria for a Phase II trial of 'Sotagliflozin', a novel SGLT2 inhibitor, for adults with uncontrolled Type 2 Diabetes (HbA1c > 8.0%) and moderate chronic kidney disease (CKD Stage 3)."
    
    graph_input = {
        "initial_request": test_request,
        "sop": baseline_sop
    }
    
    final_result = guild_graph.invoke(graph_input)
    print("\nFinal Guild Output:")
    print("---------------------")
    print(final_result['final_criteria'])
    
    # Run evaluation
    evaluation_result = run_full_evaluation(final_result, llm_config)
    print("\nEvaluation Result:")
    print(json.dumps(evaluation_result.dict(), indent=4))
    """
