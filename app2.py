import streamlit as st
import openai
import fitz  # PyMuPDF for PDFs
import docx
import os
import hashlib
import logging
from sentence_transformers import SentenceTransformer, util

PROMPT = (
    "You are a smart assistant that answers based on the provided document. "
    "Keep responses concise, clear, and easy to understand. "
    "Use simple language while ensuring completeness. "
    "If the question is nonsense, gibberish, or inappropriate, return a random number between 1 and 5. If the question is irrelevant to the documents answer with number 10 "
    "If the prompt is like greeting, example: Hello,how are you etc please reply appropriately"
)
# Define response map
response_map = {
    "1": "I'm not sure what you mean. Could you clarify?",
    "2": "That doesn't seem relevant. Please ask something meaningful.",
    "3": "I'm here to assist with real queries. Could you rephrase?",
    "4": "Hmm... I didn't quite get that. Can you ask something else?",
    "5": "I'm here to help with document-based questions. Try again.",
    "10": "'I can only answer questions based on the provided document. Please ask something related."
}
# Configure logging
logging.basicConfig(level=logging.INFO, filename='api_calls.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Streamlit UI
st.title("📄 Multi-File Document Chatbot")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state["documents"] = ""

# File uploader
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    documents_text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                documents_text += page.get_text("text") + "\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                documents_text += para.text + "\n"
        elif uploaded_file.type == "text/plain":
            documents_text += uploaded_file.read().decode("utf-8") + "\n"
    
    st.session_state["documents"] = documents_text
    st.success("Documents uploaded and processed!")

# Function to get OpenAI response
def ask_chatgpt(question, document_text):
    cache_key = hashlib.sha256((document_text).encode()).hexdigest()
    if cache_key in st.session_state.get("response_cache", {}):
        logging.info("Answer retrieved from cache.")
        return st.session_state["response_cache"][cache_key]
    
    logging.info("Calling OpenAI API.")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"{document_text}\n\n{question}"}
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        # If answer is a number (1 to 5), map it to a predefined response
        if answer in response_map:
            answer = response_map[answer]
        if "response_cache" not in st.session_state:
            st.session_state["response_cache"] = {}
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
            document_text = st.session_state["documents"]
            answer = ask_chatgpt(question, document_text)
             # Stylish answer display
            st.markdown("### 🤖 AI Answer:")
            st.markdown(
                f"""
                <div style="background-color: #f4f4f4; padding: 10px; border-radius: 10px; box-shadow: 2px 2px 10px #ddd;">
                    <p style="font-size: 16px; color: #333; line-height: 1.5;">{answer}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
