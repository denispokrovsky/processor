import streamlit as st
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
#from transformers import MarianMTModel, MarianTokenizer
import matplotlib.pyplot as plt
from pymystem3 import Mystem
import io
from rapidfuzz import fuzz
from tqdm.auto import tqdm
import time
import torch

# Initialize pymystem3 for lemmatization
mystem = Mystem()

# Set up the sentiment analyzers

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
rubert1 = pipeline("sentiment-analysis", model = "DeepPavlov/rubert-base-cased")
rubert2 = pipeline("sentiment-analysis", model = "blanchefort/rubert-base-cased-sentiment")


# Function for lemmatizing Russian text
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = []
    for word in tqdm(words, desc="Lemmatizing", unit="word"):
        lemmatized_word = ''.join(mystem.lemmatize(word))
        lemmatized_words.append(lemmatized_word)
    return ' '.join(lemmatized_words)

# Translation model for Russian to English
model_name = "Helsinki-NLP/opus-mt-ru-en"
translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

def translate(text):
    # Tokenize the input text
    inputs = translation_tokenizer(text, return_tensors="pt", truncation=True)
    
    # Calculate max_length based on input length (you may need to adjust this ratio)
    input_length = inputs.input_ids.shape[1]
    max_length = min(512, int(input_length * 1.5))
    
    # Generate translation
    translated_tokens = translation_model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the translated tokens
    translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text


# Functions for FinBERT, RoBERTa, and FinBERT-Tone with label mapping
def get_mapped_sentiment(result):
    label = result['label'].lower()
    if label in ["positive", "label_2", "pos", "pos_label"]:
        return "Positive"
    elif label in ["negative", "label_0", "neg", "neg_label"]:
        return "Negative"
    return "Neutral"

def get_rubert1_sentiment(text):
    result = rubert(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

def get_rubert2_sentiment(text):
    result = rubert(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

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
    
    original_news_count = len(df)

    # Apply fuzzy deduplication
    df = df.groupby('Объект').apply(
        lambda x: fuzzy_deduplicate(x, 'Выдержки из текста', 65)
    ).reset_index(drop=True)

    
    remaining_news_count = len(df)
    duplicates_removed = original_news_count - remaining_news_count

    st.write(f"Из {original_news_count} новостных сообщений удалены {duplicates_removed} дублирующих. Осталось {remaining_news_count}.")


    # Translate texts
    translated_texts = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_news = len(df)

    texts = df['Выдержки из текста'].tolist()

    for i, text in enumerate(df['Выдержки из текста']):
        translated_text = translate(str(lemmatize_text(text)))
        translated_texts.append(translated_text)
        progress_bar.progress((i + 1) / len(df))
        progress_text.text(f"{i + 1} из {total_news} сообщений переведено")
    
    # Perform sentiment analysis
    rubert1_results = [get_rubert1_sentiment(text) for text in texts]
    rubert2_results = [get_rubert2_sentiment(text) for text in texts]
    finbert_results = [get_finbert_sentiment(text) for text in translated_texts]
    roberta_results = [get_roberta_sentiment(text) for text in translated_texts]
    finbert_tone_results = [get_finbert_tone_sentiment(text) for text in translated_texts]
    
    # Add results to DataFrame
    df['ruBERT1'] = rubert1_results
    df['ruBERT2'] = rubert2_results
    df['FinBERT'] = finbert_results
    df['RoBERTa'] = roberta_results
    df['FinBERT-Tone'] = finbert_tone_results
    df['Translated'] = translated_texts
    
    # Reorder columns
    columns_order = ['Объект', 'ruBERT1', 'ruBERT2','FinBERT', 'RoBERTa', 'FinBERT-Tone', 'Выдержки из текста', 'Translated' ]
    df = df[columns_order]
    
    return df

def main():
    st.title("... приступим к анализу... версия 24")
    
    uploaded_file = st.file_uploader("Выбирайте Excel-файл", type="xlsx")
    
    if uploaded_file is not None:
        df = process_file(uploaded_file)
        
        st.subheader("Предпросмотр данных")
        st.write(df.head())
        
        st.subheader("Распределение окраски")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Распределение окраски по моделям")
        
        models = ['ruBERT', 'FinBERT', 'RoBERTa', 'FinBERT-Tone']
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