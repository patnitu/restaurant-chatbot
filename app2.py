import streamlit as st
import openai
import fitz  # PyMuPDF for PDFs
import docx
import os
import hashlib
import logging
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, filename='api_calls.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI API Key (using secrets management is highly recommended)
 openai.api_key = st.secrets.get("OPENAI_API_KEY") # Use get to avoid KeyError if not set
# Load the embedding model (do this only once)
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = st.session_state["embedding_model"]

# ... (rest of the text extraction functions remain the same)

# Function to get OpenAI response (improved caching and error handling)
def ask_chatgpt(question, document_text):
    cache_key = hashlib.sha256((question + document_text).encode()).hexdigest()
    if cache_key in st.session_state.get("response_cache", {}): # Use .get with default
        logging.info("Answer retrieved from cache.")
        return st.session_state["response_cache"][cache_key]

    logging.info("Calling OpenAI API.")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Consider gpt-3.5-turbo for cost-effectiveness
            messages=[
                {"role": "system", "content": "You are an assistant that answers questions based on the provided document. Be concise and use simple sentences."}, # Improved system prompt
                {"role": "user", "content": f"{document_text}\n\n{question}"}
            ],
            temperature=0.2 # Adjust temperature for more focused answers
        )
        answer = response.choices[0].message.content.strip() # .strip() removes leading/trailing whitespace
        if "response_cache" not in st.session_state:
            st.session_state["response_cache"] = {}  # Initialize if not present
        st.session_state["response_cache"][cache_key] = answer
        return answer
    except openai.error.OpenAIError as e:  # Catch specific OpenAI errors
        logging.error(f"OpenAI API Error: {e}")
        st.error(f"OpenAI API Error: {e}") # Display error to the user
        return "An error occurred with the OpenAI API. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")
        return "An unexpected error occurred. Please try again later."


# Streamlit UI
st.title("📄 Multi-File Document Chatbot")

# Initialize session state (using .get for safety)
st.session_state.setdefault("documents", "")
st.session_state.setdefault("response_cache", {})

# ... (file uploader and processing remain largely the same)

# Question input
question = st.text_input("Ask a question about the uploaded documents:")
if question:
    if not st.session_state["documents"]:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."): # Add a spinner for better UX
            # Retrieve relevant text using embeddings (improved selection)
            document_sentences = st.session_state["documents"].split(". ")
            question_embedding = embedding_model.encode(question) # No need for tensor conversion here
            doc_embeddings = embedding_model.encode(document_sentences)
            similarities = util.cos_sim(question_embedding, doc_embeddings)[0] # Use util.cos_sim directly
            top_indices = similarities.argsort(descending=True)[:min(5, len(document_sentences))]  # Limit to available sentences
            top_sentences = [document_sentences[i] for i in top_indices]
            document_text = " ".join(top_sentences)

            answer = ask_chatgpt(question, document_text)
            st.write("### 🤖 Answer:", answer)
