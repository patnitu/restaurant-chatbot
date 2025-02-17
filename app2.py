import streamlit as st
import openai
import PyPDF2

# Set OpenAI API Key (Replace with your key)
openai.api_key = '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'  # Replace with your actual OpenAI API key

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def ask_chatgpt(prompt, document_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions based on a given document."},
            {"role": "user", "content": f"Document: {document_text}\n\nQuestion: {prompt}"}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

# Streamlit UI
st.title("📄 Document-Based Chatbot")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
document_text = ""

if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded and processed successfully!")

question = st.text_input("Ask a question about the document:")
if st.button("Get Answer") and question:
    if document_text:
        answer = ask_chatgpt(question, document_text)
        st.write("**Answer:**", answer)
    else:
        st.warning("Please upload a document first.")
