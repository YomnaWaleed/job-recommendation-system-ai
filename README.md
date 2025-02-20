# 1)  AI-Powered Job Recommendation System

  This is an AI-driven job recommendation system that analyzes resumes and suggests the most relevant job positions based on skills and experience. The system utilizes **Natural Language Processing (NLP)** and **Semantic Search** to find the best job matches.

  ## 🚀 Features
  - Upload a **PDF resume** for job recommendations
  - **AI-based analysis** using `SentenceTransformer` for job matching
  - **Fast and efficient search** powered by `FAISS`
  - **TF-IDF filtering** to narrow down relevant jobs
  - **Streamlit-powered Web Interface** for easy use

  ## 🛠️ Technologies Used
  - `Python`
  - `Streamlit` (for the UI)
  - `FAISS` (for efficient job matching)
  - `Sentence Transformers` (for embedding resumes and job descriptions)
  - `PyMuPDF` (to extract text from PDF resumes)
  - `scikit-learn` (for TF-IDF filtering)

  ## 📌 How to Run
  1. Clone this repository:
    ```sh
    git clone https://github.com/YOUR_GITHUB_USERNAME/job-recommendation-system.git
    cd job-recommendation-system


# 2) Install the dependencies:
  pip install -r requirements.txt

# 3) Run the Streamlit app:
  streamlit run streamlit.py

# 4) Open the URL shown in the terminal 
  (e.g., http://localhost:8501)




# 📄 Dataset
  The system uses a CSV file (JobsFE.csv) containing job descriptions, skills, and work details. Ensure this file is present in the project folder.


# 🤖 How It Works
  Resume Processing: Extracts text from the uploaded PDF resume.
  TF-IDF Filtering: Reduces the number of job descriptions for faster and better recommendations.
  Semantic Search: Uses SentenceTransformer embeddings with FAISS for similarity search.
  Top Job Matches: The best job recommendations are displayed on the Streamlit app.


# 📌 Future Improvements
  ✅ Support for multiple resume formats (DOCX, TXT)
  ✅ Advanced customized filtering based on user preferences
  ✅ Integration with LinkedIn job postings