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
# 🔑 API CONFIGURATION (SECURE)
# ==========================================
# This looks for the key in Streamlit Secrets (Cloud) or secrets.toml (Local)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    api_status = "✅ API Connected"
else:
    # If key is missing, we don't crash, but we warn the user
    api_status = "⚠️ API Key Missing"

# ==========================================
# 🧠 DEEP LEARNING MODEL (LOCAL)
# ==========================================
@st.cache_resource
def load_deep_learning_model():
    """
    Loads the CLIP (Contrastive Language-Image Pre-Training) model.
    This is a Vision Transformer (ViT) that connects images to text concepts.
    """
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def classify_galaxy_deep_learning(image, model, processor):
    """
    Performs Zero-Shot Classification using Vector Similarity.
    """
    # 1. Define the Physics Concepts (Classes)
    labels = [
        "a spiral galaxy with rotating arms",
        "a smooth elliptical galaxy",
        "an edge-on galaxy disk viewed from the side",
        "an irregular galaxy with no shape",
        "two merging galaxies colliding"
    ]
    
    # Map descriptions back to simple names for the UI
    label_map = {
        "a spiral galaxy with rotating arms": "Spiral Galaxy",
        "a smooth elliptical galaxy": "Elliptical Galaxy",
        "an edge-on galaxy disk viewed from the side": "Edge-on Disk",
        "an irregular galaxy with no shape": "Irregular Galaxy",
        "two merging galaxies colliding": "Merger"
    }

    # 2. Preprocess Image & Text (The "Deep Learning" Input)
    inputs = processor(
        text=labels, 
        images=image, 
        return_tensors="pt", 
        padding=True
    )

    # 3. Forward Pass (Run the Model)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. Calculate Similarity (Logits to Probabilities)
    logits_per_image = outputs.logits_per_image  # Similarity score
    probs = logits_per_image.softmax(dim=1)      # Convert to percentage

    # 5. Format Results
    prob_values = probs.numpy()[0]
    results = {}
    for i, label_desc in enumerate(labels):
        simple_name = label_map[label_desc]
        results[simple_name] = float(prob_values[i])
        
    return results

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

    # --- SIDEBAR CONFIG ---
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.text(api_status) # Shows if the key was found

    if api_status == "⚠️ API Key Missing":
        st.sidebar.warning("Go to Streamlit Settings -> Secrets and add GEMINI_API_KEY")

    page = st.sidebar.radio("Select Module:", ["🔭 Galaxy Classifier (Deep Learning)", "📄 Research Assistant (NLP)"])

    # --- MODULE 1: GALAXY CLASSIFIER ---
    if page == "🔭 Galaxy Classifier (Deep Learning)":
        st.header("Deep Learning Morphology Classifier")
        st.info("Using **CLIP (Vision Transformer)** for Zero-Shot classification. No manual training required.")
        
        uploaded_file = st.file_uploader("Upload a Galaxy Image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Observation", use_column_width=True)
            
            with col2:
                if st.button("Analyze Structure"):
                    with st.spinner("Running Vision Transformer..."):
                        # Load Model
                        model, processor = load_deep_learning_model()
                        
                        # Run Inference
                        predictions = classify_galaxy_deep_learning(image, model, processor)
                        
                        # Sort and Get Top Result
                        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        top_class, top_score = sorted_preds[0]
                        
                        # Display Top Result
                        st.success(f"**Classification: {top_class}**")
                        st.metric("Confidence Score", f"{top_score*100:.2f}%")
                        
                        # Display Bar Chart of all probabilities
                        st.write("---")
                        st.write("**Probability Distribution:**")
                        st.bar_chart(predictions)

    # --- MODULE 2: RESEARCH ASSISTANT ---
    elif page == "📄 Research Assistant (NLP)":
        st.header("Research Paper Analysis")
        st.markdown("Uses **Large Language Models (LLM)** to synthesize astrophysics papers.")
        
        uploaded_pdf = st.file_uploader("Upload Research Paper (PDF)", type=['pdf'])
        
        if uploaded_pdf:
            if api_status == "⚠️ API Key Missing":
                st.error("Cannot analyze PDF. API Key is missing in Settings.")
            else:
                with st.spinner("Reading PDF structure..."):
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    st.success(f"Loaded paper containing {len(pdf_text.split())} words.")
                
                tab1, tab2 = st.tabs(["📝 Summarization", "💬 Q&A System"])
                
                # Sub-module: Summarizer
                with tab1:
                    detail_level = st.select_slider("Summary Detail", options=["Brief Abstract", "Key Findings", "Comprehensive Analysis"])
                    if st.button("Generate Summary"):
                        with st.spinner("Synthesizing..."):
                            try:
                                model = genai.GenerativeModel("gemini-pro")
                                prompt = f"You are an expert astrophysicist. Summarize this paper. \nLevel: {detail_level}. \nText: {pdf_text[:50000]}"
                                response = model.generate_content(prompt)
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

                # Sub-module: Q&A
                with tab2:
                    question = st.text_input("Ask a technical question about the paper:")
                    if st.button("Analyze Question") and question:
                        with st.spinner("Consulting paper..."):
                            try:
                                model = genai.GenerativeModel("gemini-pro")
                                prompt = f"Based STRICTLY on this paper, answer: {question}. \nPaper Text: {pdf_text[:50000]}"
                                response = model.generate_content(prompt)
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

