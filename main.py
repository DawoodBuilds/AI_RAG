import streamlit as st
from pypdf import PdfReader
import re, os
from ai import collection, simple_chat, chat_with_file

st.set_page_config(page_title="AI RAG", page_icon="rocket", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
        <style>
            section[data-testid='stSidebar'] {
                width: 400px !important;
            }
        </style>
""", unsafe_allow_html=True)

st.title("AI RAG")

def chunker(text: str, size: int, overlap: int):
    chunks : list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    return chunks

def clean_text(text: str):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text(file_path : str):
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    full_text = clean_text(full_text)
    return full_text

def process_file(uploaded_file): # type: ignore
    file_path = f'temp_{uploaded_file.name}' # type: ignore
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer()) # type: ignore
    
    status_text = st.empty()
    status_text.text("Extracting Text...")
    full_text = extract_text(file_path)
    chunks = chunker(full_text, 1000, 100)
    status_text.text(f"Indexing {len(chunks)} chunks...")
    ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))] # type: ignore
    collection.add(
        documents=chunks,
        ids=ids
    )
    os.remove(file_path)
    status_text.text("✅ Indexing Complete!")
    st.session_state.file_processed = True
        
with st.sidebar:
    st.header("Files")
    uploaded_file = st.file_uploader("Upload PDFs", type="pdf")
    if uploaded_file is None:
        st.session_state.file_processed = False
        st.session_state.input_q = "Ask something..."
    if uploaded_file:
        if st.button("Process File"):
            process_file(uploaded_file)
            st.session_state.input_q = "Ask about your pdf..."
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "file_processed" and "input_q" not in st.session_state:
    st.session_state.file_processed = False
    st.session_state.input_q = "Ask something..."

prompt = st.chat_input(st.session_state.input_q)

for message in st.session_state.messages: # type: ignore
    with st.chat_message(message["role"]): # type: ignore
        st.write(message["content"])

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt}) # type: ignore
    with st.chat_message("assistant"):
        with st.spinner("Thinking...", show_time=True):
            if st.session_state.file_processed:
                answer = chat_with_file(prompt)
            else:
                answer = simple_chat(prompt)
            if answer is None:
                answer = "I'm sorry, I encountered an error connecting to the AI."
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer}) # type: ignore