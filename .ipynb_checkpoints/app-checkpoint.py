import streamlit as st
import pandas as pd
import joblib
import os
import re
import string

st.set_page_config(
    page_title="Veritas | Fake News Detector",
    page_icon="📰",
    layout="wide"
)

MODEL_FILE = "fake_news_pipeline.pkl"

# Check if model artifact exists
if not os.path.exists(MODEL_FILE):
    st.error(f"Missing '{MODEL_FILE}'. Please run the training cells in Jupyter Notebook first to generate this file.")
    st.stop()

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_FILE)

pipeline = load_pipeline()

def clean_headline(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

# App Header
st.title("📰 Veritas News Credibility Detector")
st.markdown("Enter a news headline below to assess its authenticity and stylistic patterns.")
st.divider()

col_input, col_result = st.columns([3, 2], gap="large")

with col_input:
    st.subheader("Input Headline")
    user_text = st.text_area(
        "News Title / Headline:",
        height=150,
        placeholder="Enter or paste the headline here..."
    )
    
    col_eval, col_clear = st.columns([3, 1])
    with col_eval:
        btn_eval = st.button("Evaluate Credibility", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.rerun()

with col_result:
    st.subheader("Credibility Breakdown")
    
    if btn_eval:
        raw_text = user_text.strip()
        if raw_text:
            # Prepare feature frame
            input_df = pd.DataFrame([{
                'cleaned_title': clean_headline(raw_text),
                'caps_ratio': sum(1 for c in raw_text if c.isupper()) / (len(raw_text) + 1e-5),
                'exclamations': raw_text.count('!'),
                'questions': raw_text.count('?'),
                'char_count': len(raw_text),
                'word_count': len(raw_text.split())
            }])
            
            probs = pipeline.predict_proba(input_df)[0]
            fake_score = probs[0] * 100
            real_score = probs[1] * 100
            
            if real_score >= 50.0:
                st.success(f"### ✓ Predicted: Real News ({real_score:.1f}% confidence)")
                st.write("Stylistic and lexical patterns align with authentic reporting standards.")
            else:
                st.error(f"### ⚠ Predicted: Fake News ({fake_score:.1f}% confidence)")
                st.write("Contains linguistic features and sensationalist markers typical of disinformation.")
            
            st.write("---")
            st.write(f"**Authenticity Score (Real):** {real_score:.1f}%")
            st.progress(int(real_score))
            
            st.write(f"**Disinformation Score (Fake):** {fake_score:.1f}%")
            st.progress(int(fake_score))
        else:
            st.warning("Please type a headline before evaluating.")
    else:
        st.info("Awaiting headline input. Enter text on the left and click **Evaluate Credibility**.")