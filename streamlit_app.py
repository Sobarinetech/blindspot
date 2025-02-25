import streamlit as st
import google.generativeai as genai
import requests
import json
import supabase
from io import StringIO
from PyPDF2 import PdfReader

# Configure the API keys securely using Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize Supabase connection
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_API_KEY"]
supabase_client = supabase.create_client(supabase_url, supabase_key)

# App Title and Description
st.title("AI-Powered Legal Assistant")
st.write("Search, summarize, and analyze legal content powered by Google Search, Gemini AI, and Supabase.")

# Step 1: Enter Search Query
st.subheader("Step 1: Enter your legal query or document reference")
query = st.text_input("What would you like to search or analyze?", placeholder="e.g., 'Intellectual Property Law'")

# Function to handle web search
def search_web(query):
    """Searches the web using Google Custom Search API and returns results."""
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": st.secrets["GOOGLE_API_KEY"],
        "cx": st.secrets["GOOGLE_SEARCH_ENGINE_ID"],
        "q": query,
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        st.error(f"Search API Error: {response.status_code} - {response.text}")
        return []

# Function to summarize content using Gemini AI
def summarize_text(text):
    """Generates a summary of the legal text using Gemini AI."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Please summarize the following legal text for clarity and understanding:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()

# Function to extract documents from Supabase Storage
def fetch_legal_documents():
    """Fetches legal documents from Supabase storage."""
    storage = supabase_client.storage()
    documents = storage.from_("legal_documents").list()
    return [doc['name'] for doc in documents]

# Function to process selected document from Supabase
def get_legal_document(doc_name):
    """Retrieve the content of a selected legal document."""
    storage = supabase_client.storage()
    file = storage.from_("legal_documents").download(doc_name)
    file_content = file.decode("utf-8")  # Assuming the document is in text format
    return file_content

# Function to extract text from PDF (if applicable)
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyPDF2"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to limit text input to a manageable size
def limit_text_for_ai(text, max_length=1000):
    """Limits text to the specified number of characters."""
    if len(text) > max_length:
        return text[:max_length]  # Truncate to max_length
    return text

# Step 2: Button to perform a legal search
if query.strip():
    if st.button("Search Legal Information"):
        with st.spinner("Searching the web for related legal content... Please wait!"):
            try:
                search_results = search_web(query)
                
                if search_results:
                    st.subheader("Step 3: Legal Information Found Online")
                    for result in search_results[:3]:  # Show only the top 3 results
                        with st.expander(result['title']):
                            st.write(f"**Source:** [{result['link']}]({result['link']})")
                            st.write(f"**Snippet:** {result['snippet'][:150]}...")  # Shortened snippet
                            st.write("---")

                else:
                    st.success("No results found for your query. Consider uploading legal documents for analysis.")

            except Exception as e:
                st.error(f"Error performing search: {e}")

# Step 3: Upload Legal Document for Summarization or Analysis
st.subheader("Step 4: Upload Legal Document (PDF or Text)")
uploaded_file = st.file_uploader("Upload a legal document (e.g., law acts, contracts, etc.)", type=["txt", "pdf"])

if uploaded_file is not None:
    # Handle PDF files by extracting text
    if uploaded_file.type == "application/pdf":
        document_content = extract_text_from_pdf(uploaded_file)
    else:
        # Handle plain text files
        document_content = uploaded_file.getvalue().decode("utf-8")

    # Limit the content to the first 1,000 characters for processing
    document_content = limit_text_for_ai(document_content)

    # Show the content of the uploaded document for review
    st.subheader("Uploaded Document Preview")
    st.write(document_content[:500] + " ...")  # Preview first 500 characters

    # Summarize document using Gemini AI
    if st.button("Summarize Document"):
        with st.spinner("Summarizing the document... Please wait!"):
            try:
                summary = summarize_text(document_content)
                st.subheader("Document Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error summarizing document: {e}")

# Step 4: View Stored Legal Documents from Supabase
st.subheader("Step 5: View Stored Legal Documents")
documents = fetch_legal_documents()

if documents:
    selected_doc = st.selectbox("Select a legal document to view", documents)
    if selected_doc:
        doc_content = get_legal_document(selected_doc)
        st.subheader(f"Content of {selected_doc}")
        st.write(doc_content[:500] + " ...")  # Preview first 500 characters
        
        # Option to summarize the document
        if st.button(f"Summarize {selected_doc}"):
            summary = summarize_text(doc_content)
            st.subheader(f"Summary of {selected_doc}")
            st.write(summary)

else:
    st.write("No legal documents found in the Supabase storage. Please upload some.")

# Option to clear the input and reset the app
if st.button("Clear All Input"):
    st.session_state.clear()  # Reset the app state
    st.experimental_rerun()  # Reload the app state
