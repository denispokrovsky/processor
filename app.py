import streamlit as st
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
#from transformers import MarianMTModel, MarianTokenizer
import matplotlib.pyplot as plt
from pymystem3 import Mystem
import io
from rapidfuzz import fuzz

# Initialize pymystem3 for lemmatization
mystem = Mystem()

# Set up the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function for lemmatizing Russian text
def lemmatize_text(text):
    lemmatized_text = ''.join(mystem.lemmatize(text))
    return lemmatized_text

# Translation model for Russian to English
model_name = "Helsinki-NLP/opus-mt-ru-en"
translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

def translate(text):
    return translator(text)[0]['translation_text']

# Function for VADER sentiment analysis with label mapping
def get_vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    return "Neutral"

# Functions for FinBERT, RoBERTa, and FinBERT-Tone with label mapping
def get_mapped_sentiment(result):
    label = result['label'].lower()
    if label in ["positive", "label_2", "pos", "pos_label"]:
        return "Positive"
    elif label in ["negative", "label_0", "neg", "neg_label"]:
        return "Negative"
    return "Neutral"

def get_finbert_sentiment(text):
    result = finbert(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

def get_roberta_sentiment(text):
    result = roberta(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

def get_finbert_tone_sentiment(text):
    result = finbert_tone(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

#Fuzzy filter out similar news for the same NER
def fuzzy_deduplicate(df, column, threshold=65):
    seen_texts = []
    indices_to_keep = []
    for i, text in enumerate(df[column]):
        if pd.isna(text):
            indices_to_keep.append(i)
            continue
        text = str(text)
        if not seen_texts or all(fuzz.ratio(text, seen) < threshold for seen in seen_texts):
            seen_texts.append(text)
            indices_to_keep.append(i)
    return df.iloc[indices_to_keep]


def process_file(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    
    # Apply fuzzy deduplication
    df = df.groupby('Объект').apply(lambda x: fuzzy_deduplicate(x, 'Выдержки из текста', 65)).reset_index(drop=True)
    
    # Translate texts
    translated_texts = []
    progress_bar = st.progress(0)
    for i, text in enumerate(df['Выдержки из текста']):
        translated_text = translate(str(text))
        translated_texts.append(translated_text)
        progress_bar.progress((i + 1) / len(df))
    
    # Perform sentiment analysis
    vader_results = [get_vader_sentiment(text) for text in translated_texts]
    finbert_results = [get_finbert_sentiment(text) for text in translated_texts]
    roberta_results = [get_roberta_sentiment(text) for text in translated_texts]
    finbert_tone_results = [get_finbert_tone_sentiment(text) for text in translated_texts]
    
    # Add results to DataFrame
    df['VADER'] = vader_results
    df['FinBERT'] = finbert_results
    df['RoBERTa'] = roberta_results
    df['FinBERT-Tone'] = finbert_tone_results
    
    # Reorder columns
    columns_order = ['Объект', 'VADER', 'FinBERT', 'RoBERTa', 'FinBERT-Tone', 'Выдержки из текста']
    df = df[columns_order]
    
    return df

def main():
    st.title("... приступим к анализу...")
    
    uploaded_file = st.file_uploader("ВЫБИРАЙТЕ EXCEL-файл", type="xlsx")
    
    if uploaded_file is not None:
        df = process_file(uploaded_file)
        
        st.subheader("Предпросмотр данных")
        st.write(df.head())
        
        st.subheader("Распределение окраски")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Распределение окраски по моделям")
        
        models = ['VADER', 'FinBERT', 'RoBERTa', 'FinBERT-Tone']
        for i, model in enumerate(models):
            ax = axs[i // 2, i % 2]
            sentiment_counts = df[model].value_counts()
            sentiment_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"{model} Sentiment")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Offer download of results
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        st.download_button(
            label="Хотите загрузить результат? Вот он",
            data=output,
            file_name="sentiment_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()