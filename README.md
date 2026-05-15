# AI Resume Sorter

An AI-powered Resume Classification web application built using Streamlit and Google's Gemini Embedding Model.

The application extracts text from uploaded PDF resumes, generates embeddings using Gemini AI, compares them with predefined job role embeddings using cosine similarity, and predicts the most suitable job role.

---

# Features

- Upload Resume PDFs
- Extract text from resumes
- Generate semantic embeddings using Gemini AI
- Compare resume similarity with multiple job roles
- Predict best matching role
- Display similarity scores for all roles
- Interactive Streamlit UI

---

# Tech Stack

- Python
- Streamlit
- Google Gemini API
- NumPy
- pdfplumber

---

# Supported Job Roles

- Software Development Engineer
- Data Scientist
- Machine Learning Engineer
- Frontend Developer
- DevOps Engineer
- Cybersecurity Analyst
- Product Manager

---

# Project Structure

```bash
Resume_Shortner/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
```

---

# Installation

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Resume_Shortner
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv env
env\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Setup Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Get API key from:

https://aistudio.google.com/app/apikey

---

# Run Application

```bash
streamlit run app.py
```

---

# How It Works

## Step 1
User uploads a PDF resume.

## Step 2
Text is extracted using pdfplumber.

## Step 3
Gemini embedding model converts text into embeddings.

## Step 4
Cosine similarity compares resume embeddings with predefined role embeddings.

## Step 5
The application predicts the most suitable role.

---

# Embedding Model Used

```python
model="gemini-embedding-2"
```

---

# Similarity Algorithm

Cosine Similarity is used to measure semantic similarity between embeddings.

Formula:

```python
similarity = dot_product / (magnitude1 * magnitude2)
```

---

# Future Improvements

- Resume scoring system
- Resume summarization
- Skill extraction
- Job recommendation system
- Resume ranking dashboard
- Multi-PDF support
- Database integration

---

# Security Notes

- Never upload `.env` file
- Keep API keys private
- Use Streamlit Secrets when deploying

---

# Deployment

This project can be deployed using:

- Streamlit Community Cloud
- Render
- Railway

---

# Author

Aman Kashyap

```
