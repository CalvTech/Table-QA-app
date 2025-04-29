import streamlit as st
import pandas as pd
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

# Pagina instellingen
st.set_page_config(page_title="📊 Slimme Tabel Beantwoorder (TAPAS)", page_icon="🤖", layout="centered")

st.title("📊 Slimme Tabel Beantwoorder (TAPAS)")
st.write("Upload een CSV-bestand of gebruik de voorbeeldtabel hieronder. Stel een vraag over de gegevens, en het model zal antwoorden!")

# Model laden
@st.cache_resource
def load_model():
    model_name = "google/tapas-base-finetuned-wtq"
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("table-question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

# Tabel invoer
st.subheader("📥 Stap 1: Upload of gebruik een tabel")

uploaded_file = st.file_uploader("Upload een CSV-bestand", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Geüploade tabel:")
    st.dataframe(df)
else:
    st.info("ℹ️ Geen CSV geüpload — voorbeeldtabel wordt gebruikt.")
    df = pd.DataFrame({
        "Land": ["Nederland", "België", "Duitsland"],
        "Omzet 2022": ["€10M", "€7M", "€20M"],
        "Omzet 2023": ["€12M", "€8M", "€22M"]
    })
    st.dataframe(df)

# Vraag invoer
st.subheader("❓ Stap 2: Stel je vraag over de tabel")
question = st.text_input("Bijvoorbeeld: 'Wat is de omzet van België in 2023?'")

# Antwoord tonen
if st.button("🔍 Beantwoord vraag"):
    if question.strip():
        with st.spinner("🤔 Model denkt na..."):
            result = qa_pipeline(table=df, query=question)
        st.success("✅ Antwoord:")
        st.write(result['answer'])
    else:
        st.error("⚠️ Stel eerst een vraag.")
