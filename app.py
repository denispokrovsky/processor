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
import torch
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from sentiment_decorators import sentiment_analysis_decorator
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize pymystem3 for lemmatization
mystem = Mystem()

# Set up the sentiment analyzers
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
rubert1 = pipeline("sentiment-analysis", model="DeepPavlov/rubert-base-cased")
rubert2 = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# Translation model for Russian to English
model_name = "Helsinki-NLP/opus-mt-ru-en"
translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

def init_langchain_llm():
    pipe = pipeline("text-generation", model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def estimate_impact(llm, news_text):
    template = """
    Analyze the following news piece and estimate its monetary impact in Russian rubles for the next 6 months. 
    If a monetary estimate is not possible, categorize the impact as "Значительный", "Незначительный", or "Неопределенный".
    Also provide a short reasoning (max 100 words) for your assessment.

    News: {news}

    Estimated Impact:
    Reasoning:
    """
    prompt = PromptTemplate(template=template, input_variables=["news"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(news=news_text)
    
    impact, reasoning = response.split("Reasoning:")
    impact = impact.strip()
    reasoning = reasoning.strip()
    
    return impact, reasoning

def process_file_with_llm(df, llm):
    df['LLM_Impact'] = ''
    df['LLM_Reasoning'] = ''
    
    for index, row in df.iterrows():
        if any(row[model] in ['Negative', 'Positive'] for model in ['FinBERT', 'RoBERTa', 'FinBERT-Tone']):
            impact, reasoning = estimate_impact(llm, row['Translated'])  # Use translated text
            df.at[index, 'LLM_Impact'] = impact
            df.at[index, 'LLM_Reasoning'] = reasoning
    
    return df

def create_output_file_with_llm(df, uploaded_file, analysis_df):
    wb = load_workbook("sample_file.xlsx")
    
    # Update 'Сводка' sheet
    summary_df = pd.DataFrame({
        'Объект': df['Объект'].unique(),
        'Всего новостей': df.groupby('Объект').size(),
        'Отрицательные': df[df[['FinBERT', 'RoBERTa', 'FinBERT-Tone']].eq('Negative').any(axis=1)].groupby('Объект').size(),
        'Положительные': df[df[['FinBERT', 'RoBERTa', 'FinBERT-Tone']].eq('Positive').any(axis=1)].groupby('Объект').size(),
        'Impact': df.groupby('Объект')['LLM_Impact'].agg(lambda x: x.value_counts().index[0] if x.any() else 'Неопределенный')
    })
    ws = wb['Сводка']
    for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=False), start=4):
        for c_idx, value in enumerate(row, start=5):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Update 'Значимые' sheet
    significant_data = []
    for _, row in df.iterrows():
        if any(row[model] in ['Negative', 'Positive'] for model in ['FinBERT', 'RoBERTa', 'FinBERT-Tone']):
            sentiment = 'Negative' if any(row[model] == 'Negative' for model in ['FinBERT', 'RoBERTa', 'FinBERT-Tone']) else 'Positive'
            significant_data.append([row['Объект'], 'релевантен', sentiment, row['LLM_Impact'], row['Заголовок'], row['Выдержки из текста']])
    
    ws = wb['Значимые']
    for r_idx, row in enumerate(significant_data, start=3):
        for c_idx, value in enumerate(row, start=3):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Update 'Анализ' sheet
    analysis_df['LLM_Reasoning'] = df['LLM_Reasoning']
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

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output

# ... (keep other functions as they are)

def main():
    st.title("... приступим к анализу... версия 45")
    
    # Initialize session state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'analysis_df' not in st.session_state:
        st.session_state.analysis_df = None
    if 'llm_analyzed' not in st.session_state:
        st.session_state.llm_analyzed = False
    
    uploaded_file = st.file_uploader("Выбирайте Excel-файл", type="xlsx")
    
    if uploaded_file is not None and st.session_state.processed_df is None:
        start_time = time.time()
        
        st.session_state.processed_df = process_file(uploaded_file)
        st.session_state.analysis_df = create_analysis_data(st.session_state.processed_df)
        
        st.subheader("Предпросмотр данных")
        st.write(st.session_state.processed_df.head())
        
        st.subheader("Распределение окраски")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Распределение окраски по моделям")
        
        models = ['ruBERT2','FinBERT', 'RoBERTa', 'FinBERT-Tone']
        for i, model in enumerate(models):
            ax = axs[i // 2, i % 2]
            sentiment_counts = st.session_state.processed_df[model].value_counts()
            sentiment_counts.plot(kind='bar', ax=ax)
            ax.set_title(f"{model} Sentiment")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Анализ")
        st.dataframe(st.session_state.analysis_df)
        
        output = create_output_file(st.session_state.processed_df, uploaded_file, st.session_state.analysis_df)     
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_elapsed_time(elapsed_time)
        st.success(f"Обработка завершена за {formatted_time}.")

        st.download_button(
            label="Скачать результат анализа новостей",
            data=output,
            file_name="результат_анализа_новостей.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.session_state.processed_df is not None and not st.session_state.llm_analyzed:
        if st.button("Что скажет нейросеть?"):
            st.info("Анализ нейросетью начался. Это может занять некоторое время...")
            llm = init_langchain_llm()
            df_with_llm = process_file_with_llm(st.session_state.processed_df, llm)
            output_with_llm = create_output_file_with_llm(df_with_llm, uploaded_file, st.session_state.analysis_df)
            st.success("Анализ нейросетью завершен!")
            st.session_state.llm_analyzed = True
            st.session_state.output_with_llm = output_with_llm

    if st.session_state.llm_analyzed:
        st.download_button(
            label="Скачать результат анализа с оценкой нейросети",
            data=st.session_state.output_with_llm,
            file_name="результат_анализа_с_нейросетью.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()