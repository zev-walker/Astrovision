# 🌌 AstroVision

An AI-powered astronomy tool that combines a **zero-shot deep learning vision model** (CLIP) for galaxy morphology classification with a **large language model** (Google Gemini) for astronomy research paper analysis.

---

## 🔭 Modules

### 1. Galaxy Morphology Classifier
Upload a galaxy image and AstroVision uses OpenAI's CLIP Vision Transformer to classify it — with no task-specific training required (zero-shot).

**Classifies into 5 morphological types:**
- Spiral Galaxy
- Elliptical Galaxy
- Edge-on Disk
- Irregular Galaxy
- Merger

Returns a confidence score and a probability bar chart for all classes.

### 2. Research Paper Assistant
Upload an astronomy research paper (PDF) and interact with it using Google Gemini.

**Features:**
- **Summarization** — Choose from three detail levels: Brief Abstract, Key Findings, or Comprehensive Analysis
- **Q&A System** — Ask technical questions and get answers grounded strictly in the uploaded paper

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| Vision Model | OpenAI CLIP (`clip-vit-base-patch32`) via HuggingFace Transformers |
| Deep Learning Backend | PyTorch |
| LLM | Google Gemini API (auto-selects best available model at runtime) |
| PDF Processing | PyPDF2 |
| Image Processing | Pillow (PIL) |

---

## 🤖 Gemini Model Auto-Selection

The app automatically queries the Gemini API at startup to find the best available model. It checks in this priority order:

1. `gemini-1.5-flash`
2. `gemini-1.5-flash-001`
3. `gemini-pro`
4. `gemini-1.0-pro`
5. Falls back to the first available model if none of the above are found

This prevents hardcoded model name failures.

---

## 🚀 Running the App

### Prerequisites

```bash
pip install -r requirements.txt
```

### API Key Setup

Create a `.streamlit/secrets.toml` file in your project folder:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

### Run Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📝 Notes

- CLIP model (`~600 MB`) downloads automatically on first run and is cached after that
- PDF text is capped at 50,000 characters when sent to Gemini
- The app uses a space-themed dark UI built with custom CSS

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io)
- [HuggingFace Transformers](https://huggingface.co/openai/clip-vit-base-patch32)
- [Google Gemini API](https://ai.google.dev)
- [PyTorch](https://pytorch.org)
