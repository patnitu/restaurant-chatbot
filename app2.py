import streamlit as st
import openai
import fitz  # PyMuPDF for PDFs
import docx
import os
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='api_calls.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Streamlit UI
st.title("📄 Multi-File Document Chatbot")

# Initialize session state
st.session_state.setdefault("documents", "")
st.session_state.setdefault("response_cache", {})

def extract_relevant_text(question, document_text):
    """Extracts relevant sentences from document text based on keyword matching."""
    question_words = set(question.lower().split())
    sentences = document_text.split(". ")
    relevant_sentences = [s for s in sentences if question_words & set(s.lower().split())]
    return " ".join(relevant_sentences[:5])  # Return up to 5 relevant sentences

def ask_chatgpt(question, document_text):
    cache_key = hashlib.sha256((question + document_text).encode()).hexdigest()
    if cache_key in st.session_state["response_cache"]:
        logging.info("Answer retrieved from cache.")
        return st.session_state["response_cache"][cache_key]

    logging.info("Calling OpenAI API.")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are an assistant that answers questions based on the provided document."},
                {"role": "user", "content": f"{document_text}\n\n{question}"}
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        st.session_state["response_cache"][cache_key] = answer
        return answer
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API Error: {e}")
        st.error(f"OpenAI API Error: {e}")
        return "An error occurred with the OpenAI API. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")
        return "An unexpected error occurred. Please try again later."

# Question input
question = st.text_input("Ask a question about the uploaded documents:")
if question:
    if not st.session_state["documents"]:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            document_text = extract_relevant_text(question, st.session_state["documents"])
            answer = ask_chatgpt(question, document_text)
            st.write("### 🤖 Answer:", answer)
