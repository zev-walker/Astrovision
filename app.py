"""
🌌 AstroVision - Deep Learning Galaxy Morphology & Research Assistant
Powered by Zero-Shot Vision Transformers (CLIP) and LLMs (Gemini)
"""
import streamlit as st
import torch
import google.generativeai as genai
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import PyPDF2
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AstroVision",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Space" theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    h1, h2, h3 {
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔑 API & MODEL AUTO-CONFIGURATION
# ==========================================
# This function automatically finds a working model name to fix the 404 error
@st.cache_resource
def get_valid_model_name(api_key):
    try:
        genai.configure(api_key=api_key)
        # Ask Google for list of all models
        models = list(genai.list_models())
        
        # Priority list (We prefer Flash, then Pro)
        preferences = ["gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-pro", "gemini-1.0-pro"]
        
        available_names = [m.name.replace("models/", "") for m in models if 'generateContent' in m.supported_generation_methods]
        
        # 1. Check if any of our preferred models exist
        for pref in preferences:
            if pref in available_names:
                return pref
        
        # 2. Fallback: Just take the first one that works
        if available_names:
            return available_names[0]
            
        return None
    except Exception as e:
        return None

# Check for Key in Secrets
active_key = None
valid_model_name = None

if "GEMINI_API_KEY" in st.secrets:
    active_key = st.secrets["GEMINI_API_KEY"]
    valid_model_name = get_valid_model_name(active_key)
    
    if valid_model_name:
        api_status = f"✅ Connected ({valid_model_name})"
    else:
        api_status = "⚠️ Key found, but no models available"
else:
    api_status = "⚠️ API Key Missing"

# ==========================================
# 🧠 DEEP LEARNING MODEL (LOCAL)
# ==========================================
@st.cache_resource
def load_deep_learning_model():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def classify_galaxy_deep_learning(image, model, processor):
    labels = [
        "a spiral galaxy with rotating arms",
        "a smooth elliptical galaxy",
        "an edge-on galaxy disk viewed from the side",
        "an irregular galaxy with no shape",
        "two merging galaxies colliding"
    ]
    label_map = {
        "a spiral galaxy with rotating arms": "Spiral Galaxy",
        "a smooth elliptical galaxy": "Elliptical Galaxy",
        "an edge-on galaxy disk viewed from the side": "Edge-on Disk",
        "an irregular galaxy with no shape": "Irregular Galaxy",
        "two merging galaxies colliding": "Merger"
    }
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).numpy()[0]
    
    return {label_map[labels[i]]: float(probs[i]) for i in range(len(labels))}

# ==========================================
# 📄 NLP & PDF PROCESSING
# ==========================================
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ==========================================
# 🚀 MAIN APP LOGIC
# ==========================================
def main():
    st.title("🌌 AstroVision: Deep Learning for Astronomy")
    st.markdown("### 🔭 Automated Galaxy Morphology & Research Analysis")

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.text(api_status)
    
    if "Missing" in api_status:
        st.sidebar.warning("Add GEMINI_API_KEY to Streamlit Secrets.")

    page = st.sidebar.radio("Select Module:", ["🔭 Galaxy Classifier (Deep Learning)", "📄 Research Assistant (NLP)"])

    # --- MODULE 1: GALAXY CLASSIFIER ---
    if page == "🔭 Galaxy Classifier (Deep Learning)":
        st.header("Deep Learning Morphology Classifier")
        st.info("Using **CLIP (Vision Transformer)** for Zero-Shot classification.")
        
        uploaded_file = st.file_uploader("Upload a Galaxy Image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Observation", use_column_width=True)
            with col2:
                if st.button("Analyze Structure"):
                    with st.spinner("Running Vision Transformer..."):
                        model, processor = load_deep_learning_model()
                        predictions = classify_galaxy_deep_learning(image, model, processor)
                        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        top_class, top_score = sorted_preds[0]
                        
                        st.success(f"**Classification: {top_class}**")
                        st.metric("Confidence Score", f"{top_score*100:.2f}%")
                        st.bar_chart(predictions)

    # --- MODULE 2: RESEARCH ASSISTANT ---
    elif page == "📄 Research Assistant (NLP)":
        st.header("Research Paper Analysis")
        uploaded_pdf = st.file_uploader("Upload Research Paper (PDF)", type=['pdf'])
        
        if uploaded_pdf:
            if not valid_model_name:
                st.error("Cannot analyze. API Key missing or no valid models found.")
            else:
                with st.spinner("Reading PDF structure..."):
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    st.success(f"Loaded paper containing {len(pdf_text.split())} words.")
                
                tab1, tab2 = st.tabs(["📝 Summarization", "💬 Q&A System"])
                
                with tab1:
                    detail_level = st.select_slider("Summary Detail", options=["Brief Abstract", "Key Findings", "Comprehensive Analysis"])
                    if st.button("Generate Summary"):
                        with st.spinner(f"Synthesizing using {valid_model_name}..."):
                            try:
                                # USE THE AUTO-DETECTED MODEL NAME HERE
                                model = genai.GenerativeModel(valid_model_name)
                                prompt = f"You are an expert astrophysicist. Summarize this paper. \nLevel: {detail_level}. \nText: {pdf_text[:50000]}"
                                response = model.generate_content(prompt)
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

                with tab2:
                    question = st.text_input("Ask a technical question about the paper:")
                    if st.button("Analyze Question") and question:
                        with st.spinner("Consulting paper..."):
                            try:
                                # USE THE AUTO-DETECTED MODEL NAME HERE
                                model = genai.GenerativeModel(valid_model_name)
                                prompt = f"Based STRICTLY on this paper, answer: {question}. \nPaper Text: {pdf_text[:50000]}"
                                response = model.generate_content(prompt)
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
