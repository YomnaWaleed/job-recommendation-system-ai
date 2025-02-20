import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
from model import JobRecommendationSystem

# Load the job recommendation model
recommender = JobRecommendationSystem("JobsFE.csv")

st.title("AI-Powered Job Recommendation System")

st.write(
    "Upload your resume as a PDF file, and get 20 job recommendations tailored to you!"
)

# File uploader for PDF resume
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()


resume_text = ""

if uploaded_file:
    with st.spinner("Extracting text from your resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)

if st.button("Recommend Jobs"):
    if resume_text:
        with st.spinner("Analyzing your resume and finding best job matches..."):
            recommendations = recommender.recommend_jobs(resume_text, top_n=20)
            job_results = recommendations["recommended_jobs"]

        st.success(f"Found {len(job_results)} job recommendations for you!")

        # Display recommended jobs
        for i, job in enumerate(job_results, start=1):
            st.subheader(f"Job {i}: {job['position']}")
            st.write(f"**Company:** {job['workplace']}")
            st.write(f"**Mode:** {job['working_mode']}")
            st.write(f"**Duties:** {job['job_role_and_duties']}")
            st.write(f"**Required Skills:** {job['requisite_skill']}")
            st.write("---")
    else:
        st.warning("Please upload a valid PDF resume before proceeding.")
