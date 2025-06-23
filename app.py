import streamlit as st
import pdfplumber
from transformers import pipeline

@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    generator = pipeline("text2text-generation", model="google/flan-t5-large")
    return classifier, generator

classifier, generator = load_models()

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text.strip()

st.title("📄 Resume Classifier + AI Feedback Generator")

uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted!")

    st.subheader("📜 Extracted Resume Preview")
    st.text_area("Resume Text", resume_text[:2000], height=200)

    if st.button("🔎 Analyze Resume"):
        with st.spinner("🤖 Classifying job role..."):
            job_roles = [
                "Software Engineer", "Data Analyst", "Data Scientist",
                "Backend Developer", "UI/UX Designer"
            ]
            result = classifier(resume_text, job_roles)
            predicted_role = result["labels"][0]
            st.success(f"🎯 Predicted Role: **{predicted_role}**")

        with st.spinner("💡 Generating Feedback..."):
            prompt = f"Provide resume improvement suggestions for a {predicted_role} based on the following:\n{resume_text[:1000]}"
            feedback = generator(prompt, max_length=200)[0]['generated_text']
            st.subheader("📝 AI Feedback")
            st.write(feedback)
