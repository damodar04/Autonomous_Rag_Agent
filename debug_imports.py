import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'

print("Importing dotenv...")
from dotenv import load_dotenv
print("Importing langchain_openai...")
from langchain_openai import ChatOpenAI
print("Importing langchain_community.document_loaders...")
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
print("Importing langchain_text_splitters...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Importing langchain_community.vectorstores...")
from langchain_community.vectorstores import FAISS
print("Importing duckdb...")
import duckdb
print("Importing pandas...")
import pandas as pd
print("Importing plotly...")
import plotly.graph_objects as go
print("Importing sklearn...")
from sklearn.feature_extraction.text import TfidfVectorizer
print("Imports finished successfully.")
