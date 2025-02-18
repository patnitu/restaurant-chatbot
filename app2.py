import os
os.system("pip install sentence-transformers")
import streamlit as st
import openai
import fitz  # PyMuPDF for PDFs
import docx
import hashlib
import logging
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, filename='api.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Secrets Management (Essential for API Keys)
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Load Embedding Model (Once per session)
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = st.session_state["embedding_model"]

# Text Extraction Functions (Concise and robust)
def extract_text(file):
    try:
        if file.type == "application/pdf":
            doc = fitz.open(stream=bytes(file.read()), filetype="pdf")
            return "\n".join(page.get_text("text") for page in doc)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return "\n".join(para.text for para in doc.paragraphs)
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        else:
            st.error("Unsupported file type.")  # Inform user about unsupported types
            return ""
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        st.error(f"Error processing file: {e}") # Show error to the user
        return ""

# Question Normalization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def normalize_question(question):
    question = question.lower()
    question = re.sub(r'[^\w\s]', '', question)  # Remove punctuation
    question = re.sub(r'\s+', ' ', question).strip()  # Remove extra spaces

    words = question.split()
    words = [word for word in words if word not in stop_words]  # Remove Stop words
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization

    return " ".join(words)

# OpenAI API Interaction (with enhanced error handling and prompt engineering)
def ask_chatgpt(question, document_text):
    question = question.strip().lower()
    normalized_question = normalize_question(question)
    embedding = embedding_model.encode(normalized_question)  # Get embedding
    embedding_bytes = embedding.tobytes() 
    logging.info(f"Question asked: {question}")
    cache_key = hashlib.sha256(embedding_bytes).hexdigest()
    if cache_key in st.session_state.get("response_cache", {}):
        logging.info("From Cache")
        return st.session_state["response_cache"][cache_key]

    try:
      logging.info("Calling OpenAPI API Now")
      response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # or gpt-4 if needed
            messages=[
                {"role": "system", "content": """
                You are a helpful assistant.  Answer questions concisely and in simple sentences based on the provided document. 
                If the answer is not contained in the document, say "I don't have enough information to answer that." 
                Avoid making up information.
                """},
                {"role": "user", "content": f"{document_text}\n\n{question}"}
            ],
            temperature=0.2, # Adjust for desired creativity
            max_tokens=250 # Control response length
        )
      answer = response["choices"][0]["message"]["content"]
      st.session_state.setdefault("response_cache", {})[cache_key] = answer
      logging.info("OpenAI API call successful.")
      return answer
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API Error: {e}")
        st.error(f"OpenAI API Error: {e}")
        return "An error occurred with the OpenAI API."
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred.")
        return "An unexpected error occurred."


# Streamlit UI
st.title("📄 AI Assistant")

# Session State Initialization
st.session_state.setdefault("documents", "")
st.session_state.setdefault("response_cache", {})

# File Uploader and Processing
uploaded_files = st.file_uploader("Upload documents (pdf, docx, txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        extracted_text = extract_text(file)
        if extracted_text:  # Only add if extraction was successful
            all_text += extracted_text + "\n"

    st.session_state["documents"] = all_text
    st.success("Documents processed!")



# Question Input and Answer Display
question = st.text_input("Ask a question:")
if question:
    if not st.session_state["documents"]:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
          document_sentences = st.session_state["documents"].split(". ")
          if document_sentences: # Check if document_sentences is not empty
              question_embedding = embedding_model.encode(question)
              doc_embeddings = embedding_model.encode(document_sentences)
              similarities = util.cos_sim(question_embedding, doc_embeddings)[0]
              top_indices = similarities.argsort(descending=True)[:min(5, len(document_sentences))]
              top_sentences = [document_sentences[i] for i in top_indices]
              relevant_text = " ".join(top_sentences)

              answer = ask_chatgpt(question, relevant_text)
              st.write("### 🤖 Answer:", answer)
          else:
            st.warning("No sentences found in the uploaded documents.")
