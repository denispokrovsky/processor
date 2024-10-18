import streamlit as st
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import matplotlib.pyplot as plt
from pymystem3 import Mystem
import io
from rapidfuzz import fuzz
from tqdm.auto import tqdm
import time
import torch
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Initialize pymystem3 for lemmatization
mystem = Mystem()

# Set up the sentiment analyzers

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
rubert1 = pipeline("sentiment-analysis", model = "DeepPavlov/rubert-base-cased")
rubert2 = pipeline("sentiment-analysis", model = "blanchefort/rubert-base-cased-sentiment")

def create_analysis_data(df):
    analysis_data = []
    for _, row in df.iterrows():
        if any(row[model] == 'Negative' for model in ['FinBERT', 'RoBERTa', 'FinBERT-Tone']):
            analysis_data.append([row['Объект'], 'РИСК УБЫТКА', row['Заголовок'], row['Выдержки из текста']])
    return pd.DataFrame(analysis_data, columns=['Объект', 'Тип риска', 'Заголовок', 'Текст'])

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
    result = rubert1(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

def get_rubert2_sentiment(text):
    result = rubert2(text, truncation=True, max_length=512)[0]
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
    
    required_columns = ['Объект', 'Заголовок', 'Выдержки из текста']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Error: The following required columns are missing from the input file: {', '.join(missing_columns)}")
        st.stop()
    
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
    lemmatized_texts = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_news = len(df)

    texts = df['Выдержки из текста'].tolist()

    for text in df['Выдержки из текста']: 
        lemmatized_texts.append(lemmatize_text(text))
    
    for i, text in enumerate(lemmatized_texts):
        translated_text = translate(str(text))
        translated_texts.append(translated_text)
        progress_bar.progress((i + 1) / len(df))
        progress_text.text(f"{i + 1} из {total_news} сообщений предобработано")
    
    # Perform sentiment analysis
    rubert2_results = [get_rubert2_sentiment(text) for text in texts]
    finbert_results = [get_finbert_sentiment(text) for text in translated_texts]
    roberta_results = [get_roberta_sentiment(text) for text in translated_texts]
    finbert_tone_results = [get_finbert_tone_sentiment(text) for text in translated_texts]
    
    # Create a new DataFrame with processed data
    processed_df = pd.DataFrame({
        'Объект': df['Объект'],
        'Заголовок': df['Заголовок'],  # Preserve original 'Заголовок'
        'ruBERT2': rubert2_results,
        'FinBERT': finbert_results,
        'RoBERTa': roberta_results,
        'FinBERT-Tone': finbert_tone_results,
        'Выдержки из текста': df['Выдержки из текста'],
        'Translated': translated_texts
    })
    
    return processed_df

def create_output_file(df, uploaded_file, analysis_df):
    # Load the sample file to use as a template
    wb = load_workbook("sample_file.xlsx")
    
    # Process data for 'Сводка' sheet
    entities = df['Объект'].unique()
    summary_data = []
    for entity in entities:
        entity_df = df[df['Объект'] == entity]
        total_news = len(entity_df)
        negative_news = sum((entity_df['FinBERT'] == 'Negative') | 
                            (entity_df['RoBERTa'] == 'Negative') | 
                            (entity_df['FinBERT-Tone'] == 'Negative'))
        positive_news = sum((entity_df['FinBERT'] == 'Positive') | 
                            (entity_df['RoBERTa'] == 'Positive') | 
                            (entity_df['FinBERT-Tone'] == 'Positive'))
        summary_data.append([entity, total_news, negative_news, positive_news])
    
    summary_df = pd.DataFrame(summary_data, columns=['Объект', 'Всего новостей', 'Отрицательные', 'Положительные'])
    summary_df = summary_df.sort_values('Отрицательные', ascending=False)
    
    # Write 'Сводка' sheet
    ws = wb['Сводка']
    for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=False), start=4):
        for c_idx, value in enumerate(row, start=5):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Process data for 'Значимые' sheet
    significant_data = []
    for _, row in df.iterrows():
        if any(row[model] in ['Negative', 'Positive'] for model in ['FinBERT', 'RoBERTa', 'FinBERT-Tone']):
            sentiment = 'Negative' if any(row[model] == 'Negative' for model in ['FinBERT', 'RoBERTa', 'FinBERT-Tone']) else 'Positive'
            significant_data.append([row['Объект'], sentiment, row['Заголовок'], row['Выдержки из текста']])
    
    # Write 'Значимые' sheet
    ws = wb['Значимые']
    for r_idx, row in enumerate(significant_data, start=3):
        for c_idx, value in enumerate(row, start=3):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Write 'Анализ' sheet
    ws = wb['Анализ']
    for r_idx, row in enumerate(dataframe_to_rows(analysis_df, index=False, header=False), start=4):
        for c_idx, value in enumerate(row, start=5):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Copy 'Публикации' sheet from original uploaded file
    original_df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    ws = wb['Публикации']
    for r_idx, row in enumerate(dataframe_to_rows(original_df, index=False, header=True), start=1):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Add 'Тех.приложение' sheet with processed data
    if 'Тех.приложение' not in wb.sheetnames:
        wb.create_sheet('Тех.приложение')
    ws = wb['Тех.приложение']
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Save the workbook to a BytesIO object
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    
    return output

def main():
    st.title("... приступим к анализу... версия 37+")
    
    uploaded_file = st.file_uploader("Выбирайте Excel-файл", type="xlsx")
    
    if uploaded_file is not None:
        df = process_file(uploaded_file)
        
        st.subheader("Предпросмотр данных")
        st.write(df.head())
        
        st.subheader("Распределение окраски")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Распределение окраски по моделям")
        
        models = ['ruBERT2','FinBERT', 'RoBERTa', 'FinBERT-Tone']
        for i, model in enumerate(models):
            ax = axs[i // 2, i % 2]
            sentiment_counts = df[model].value_counts()
            sentiment_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"{model} Sentiment")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
        
        plt.tight_layout()
        st.pyplot(fig)
        analysis_df = create_analysis_data(df)
        st.subheader("Анализ")
        st.dataframe(analysis_df)


        # Offer download of results
        output = create_output_file(df, uploaded_file, analysis_df)
        st.download_button(
            label="Скачать результат анализа новостей",
            data=output,
            file_name="результат_анализа_новостей.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
if __name__ == "__main__":
    main()