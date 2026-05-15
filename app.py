import os
import numpy as np
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:

    st.error(
        "Google API key not found. "
        "Add GOOGLE_API_KEY to your .env file."
    )

    st.stop()




client = genai.Client(api_key=API_KEY)




JOB_ROLES = {

    "Software Development Engineer":
    """
    Builds backend systems, APIs, scalable applications,
    databases using Python, Java, C++, Django, Flask,
    system design, OOPs, DSA and algorithms.
    """,


    "Data Scientist":
    """
    Performs data analysis, statistics, data visualization,
    predictive analytics, SQL, pandas, numpy,
    experimentation and reporting.
    """,


    "Machine Learning Engineer":
    """
    Builds machine learning pipelines, deep learning systems,
    TensorFlow, PyTorch, neural networks,
    feature engineering and MLOps.
    """,


    "Frontend Developer":
    """
    Develops responsive web interfaces using HTML,
    CSS, JavaScript, React, Next.js, Tailwind CSS,
    UI/UX principles and frontend optimization.
    """,


    "DevOps Engineer":
    """
    Handles CI/CD pipelines, Docker, Kubernetes,
    AWS cloud infrastructure, Linux servers,
    deployment automation and monitoring systems.
    """,


    "Cybersecurity Analyst":
    """
    Performs penetration testing,
    vulnerability assessment,
    SIEM monitoring, network security,
    threat analysis and incident response.
    """,


    "Product Manager":
    """
    Defines product roadmap,
    agile workflows,
    customer requirements,
    stakeholder communication,
    business strategy and planning.
    """
}


# =====================================================
# FUNCTION 1
# EXTRACT TEXT FROM PDF
# =====================================================

def extract_text_from_pdf(uploaded_file):

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text.strip()


# =====================================================
# FUNCTION 2
# GENERATE EMBEDDING
# =====================================================

def get_embedding(text):

    text = text[:8000]

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    embedding = response.embeddings[0].values

    return embedding


# =====================================================
# FUNCTION 3
# COSINE SIMILARITY
# =====================================================

def cosine_similarity(vec1, vec2):

    vec1 = np.array(vec1)

    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)

    magnitude1 = np.linalg.norm(vec1)

    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:

        return 0

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# =====================================================
# FUNCTION 4
# LOAD CATEGORY EMBEDDINGS
# =====================================================

@st.cache_resource
def load_category_embeddings():

    category_embeddings = {}

    for role, description in JOB_ROLES.items():

        embedding = get_embedding(description)

        category_embeddings[role] = embedding

    return category_embeddings


# =====================================================
# FUNCTION 5
# CLASSIFY RESUME
# =====================================================

def classify_resume(resume_text, category_embeddings):

    resume_embedding = get_embedding(resume_text)

    best_role = None

    best_score = -1

    scores = {}

    for role, embedding in category_embeddings.items():

        score = cosine_similarity(
            resume_embedding,
            embedding
        )

        scores[role] = score

        if score > best_score:

            best_score = score

            best_role = role

    return best_role, best_score, scores


# =====================================================
# STREAMLIT UI
# =====================================================

st.title("AI Resume Sorter")

st.caption(
    "Upload a resume PDF and predict the most suitable job role."
)


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

with st.spinner("Loading job role embeddings..."):

    category_embeddings = load_category_embeddings()


# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


# =====================================================
# RUN IF FILE UPLOADED
# =====================================================

if uploaded_file:

    st.success("Resume uploaded successfully!")

    if st.button("Classify Resume"):

        # =========================================
        # EXTRACT TEXT
        # =========================================

        with st.spinner("Reading PDF..."):

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

        # =========================================
        # EMPTY CHECK
        # =========================================

        if not resume_text:

            st.error(
                "No readable text found in PDF."
            )

            st.stop()

        # =========================================
        # SHOW TEXT
        # =========================================

        st.subheader("Extracted Resume Text")

        st.text_area(
            "Resume Content",
            resume_text,
            height=250
        )

        # =========================================
        # CLASSIFY
        # =========================================

        with st.spinner("Analyzing Resume..."):

            predicted_role, best_score, scores = classify_resume(
                resume_text,
                category_embeddings
            )

        

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Role",
            value=predicted_role
        )

        st.write(
            f"Best Similarity Score: {best_score:.4f}"
        )

        # =========================================
        # SHOW ALL SCORES
        # =========================================

        st.subheader("All Similarity Scores")

        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for role, score in sorted_scores:

            st.write(f"{role}: {score:.4f}")