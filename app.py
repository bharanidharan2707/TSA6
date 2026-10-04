import streamlit as st
from transformers import pipeline

# Cache the model to prevent reloading on every run
@st.cache_resource
def load_summarizer():
    # Using a fast and small model suitable for CPU
    model_name = "sshleifer/distilbart-cnn-12-6"
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Set up page configuration
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="centered")

st.title("📝 AI Text Summarizer")
st.write("Enter a long paragraph or article below, and get a short, meaningful summary.")

# Load the summarization pipeline
summarizer = load_summarizer()

# User input
text_input = st.text_area("Input Text", height=250, placeholder="Paste your text here...")

# Summarize button
if st.button("Summarize", type="primary"):
    if text_input.strip():
        with st.spinner("Generating summary (this may take a moment on the first run)..."):
            try:
                # Generate summary
                tokenizer, model = summarizer
                inputs = tokenizer(text_input, return_tensors="pt", max_length=1024, truncation=True)
                summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, do_sample=False)
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                st.subheader("Summary:")
                st.info(summary_text)
            except Exception as e:
                st.error(f"An error occurred: {e}. Ensure the text is long enough for summarization.")
    else:
        st.warning("Please enter some text before clicking summarize.")
