# 🌌 AstroVision
### Deep Learning Galaxy Morphology & Research Assistant

AstroVision is a two-module Streamlit application that combines a zero-shot Vision Transformer (CLIP) for galaxy image classification with a Gemini-powered LLM for astronomy research paper analysis.

---

## ✨ Features

### 🔭 Module 1 — Galaxy Classifier (Deep Learning)

Upload a galaxy image and AstroVision will classify it using **CLIP (`openai/clip-vit-base-patch32`)**, a zero-shot Vision Transformer from OpenAI via Hugging Face Transformers. No fine-tuning required — classification runs entirely locally via PyTorch.

**Supported galaxy types:**
- Spiral Galaxy
- Elliptical Galaxy
- Edge-on Disk
- Irregular Galaxy
- Merger (two colliding galaxies)

**Output:**
- Top predicted class with confidence score
- Bar chart of probability distribution across all 5 classes

---

### 📄 Module 2 — Research Assistant (NLP)

Upload an astronomy research paper (PDF) and interact with it using **Google Gemini**.

**Sub-features:**
- **📝 Summarization** — Generate a summary at one of three detail levels: *Brief Abstract*, *Key Findings*, or *Comprehensive Analysis*
- **💬 Q&A System** — Ask technical questions about the paper; Gemini answers strictly based on the paper's content

PDF text is extracted using `PyPDF2`. Up to 50,000 characters of the paper are sent to Gemini per request.

---

## 🛠️ Tech Stack

| Component | Library / Service |
|---|---|
| App framework | Streamlit |
| Vision model | CLIP (`openai/clip-vit-base-patch32`) via `transformers` |
| Tensor inference | PyTorch (`torch`) |
| Image handling | Pillow (`PIL`) |
| LLM (NLP) | Google Gemini (`google-generativeai`) |
| PDF parsing | PyPDF2 |

---

## ⚙️ Setup & Configuration

### 1. Clone the repository

```bash
git clone https://github.com/zev-walker/Astrovision.git
cd Astrovision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Gemini API Key

The Research Assistant requires a Google Gemini API key. AstroVision reads it from **Streamlit Secrets**.

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

> Get a free API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

The app auto-detects the best available Gemini model from your account at startup, preferring `gemini-1.5-flash` → `gemini-1.5-flash-001` → `gemini-pro` → `gemini-1.0-pro`.

> **Note:** The Galaxy Classifier runs fully locally and does **not** require the API key.

### 4. Run locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🚀 Deploying to Streamlit Cloud

1. Push your code to a public GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** and fill in:
   - **Repository:** `your-username/Astrovision`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```
5. Click **"Deploy!"** and wait 2–5 minutes.

> **First load is slow** — CLIP downloads ~600 MB of model weights on first run. Streamlit caches the model after that (`@st.cache_resource`), so subsequent loads are fast.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `⚠️ API Key Missing` shown in sidebar | Add `GEMINI_API_KEY` to Streamlit Secrets (`.streamlit/secrets.toml` locally, or the Secrets panel on Streamlit Cloud) |
| `⚠️ Key found, but no models available` | Your API key is valid but no Gemini models with `generateContent` support were returned — check your Google Cloud project quota |
| Slow first load | Normal — CLIP model weights are downloading. They are cached after the first run |
| PDF not loading / empty text | Some scanned PDFs contain no extractable text. `PyPDF2` only handles text-based PDFs |
| `app.py` not found on deploy | Ensure `app.py` and `requirements.txt` are in the **root** of the repository, not inside a subfolder |

---

## 📁 Repository Structure

```
Astrovision/
├── app.py            # Main Streamlit application (single-file)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [CLIP on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
- [Google Gemini API](https://aistudio.google.com)
- [Galaxy Zoo Dataset (Kaggle)](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)

---

**Built with ❤️ for Astronomy and AI**
