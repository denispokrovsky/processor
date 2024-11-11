import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from openpyxl.utils.dataframe import dataframe_to_rows
import io
from rapidfuzz import fuzz
import os
from openpyxl import load_workbook
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from io import StringIO, BytesIO
import sys
import contextlib
from langchain_openai import ChatOpenAI  # Updated import
import pdfkit
from jinja2 import Template
from googletrans import Translator as GoogleTranslator
import time

class TranslationSystem:
    def __init__(self, method='googletrans', llm=None):
        """
        Initialize translation system with specified method.
        
        Args:
            method (str): 'googletrans' or 'llm'
            llm: LangChain LLM instance (required if method is 'llm')
        """
        self.method = method
        self.llm = llm
        if method == 'googletrans':
            try:
                self.google_translator = GoogleTranslator()
                # Test the translator with a simple string
                self.google_translator.translate('test', src='en', dest='ru')
            except Exception as e:
                st.warning(f"Error initializing Google Translator: {str(e)}. Falling back to LLM translation.")
                self.method = 'llm'
        else:
            self.google_translator = None
        
    def translate_text(self, text, src='ru', dest='en'):
        """
        Translate text using the selected translation method.
        
        Args:
            text (str): Text to translate
            src (str): Source language code
            dest (str): Destination language code
            
        Returns:
            str: Translated text
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return text
            
        try:
            if self.method == 'googletrans' and self.google_translator:
                return self._translate_with_googletrans(text, src, dest)
            else:
                return self._translate_with_llm(text, src, dest)
        except Exception as e:
            st.warning(f"Translation error: {str(e)}. Returning original text.")
            return text
            
    def _translate_with_googletrans(self, text, src='ru', dest='en'):
        """
        Translate using googletrans library with improved error handling.
        """
        try:
            # Clean and validate input text
            text = text.strip()
            if not text:
                return text
                
            # Add delay to avoid rate limits
            time.sleep(0.5)
            
            # Attempt translation with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.google_translator.translate(text, src=src, dest=dest)
                    if result and result.text:
                        return result.text
                    raise Exception("Empty translation result")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retry
                    
            raise Exception("All translation attempts failed")
            
        except Exception as e:
            # If googletrans fails, fall back to LLM translation
            if self.llm:
                st.warning(f"Googletrans error: {str(e)}. Falling back to LLM translation.")
                return self._translate_with_llm(text, src, dest)
            raise Exception(f"Googletrans error: {str(e)}")
            
    def _translate_with_llm(self, text, src='ru', dest='en'):
        """
        Translate using LangChain LLM with improved error handling.
        """
        if not self.llm:
            raise Exception("LLM not initialized for translation")
            
        try:
            # Clean input text
            text = text.strip()
            if not text:
                return text
                
            # Prepare system message based on language direction
            if src == 'ru' and dest == 'en':
                system_msg = "You are a translator. Translate the given Russian text to English accurately and concisely."
                user_msg = f"Translate this Russian text to English: {text}"
            elif src == 'en' and dest == 'ru':
                system_msg = "You are a translator. Translate the given English text to Russian accurately and concisely."
                user_msg = f"Translate this English text to Russian: {text}"
            else:
                raise Exception(f"Unsupported language pair: {src} to {dest}")
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle different response types
            if hasattr(response, 'content'):
                translation = response.content.strip()
            elif isinstance(response, str):
                translation = response.strip()
            else:
                translation = str(response).strip()
                
            if not translation:
                raise Exception("Empty translation result")
                
            return translation
            
        except Exception as e:
            raise Exception(f"LLM translation error: {str(e)}")

def process_file(uploaded_file, model_choice, translation_method='googletrans'):
    df = None
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Публикации')
        llm = init_langchain_llm(model_choice)
        
        # Initialize translation system with chosen method
        translator = TranslationSystem(
            method=translation_method,
            llm=llm if translation_method == 'llm' else None
        )
        
        # Validate required columns
        required_columns = ['Объект', 'Заголовок', 'Выдержки из текста']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Error: The following required columns are missing: {', '.join(missing_columns)}")
            return df if df is not None else None
        
        # Deduplication
        original_news_count = len(df)
        df = df.groupby('Объект', group_keys=False).apply(
            lambda x: fuzzy_deduplicate(x, 'Выдержки из текста', 65)
        ).reset_index(drop=True)
    
        remaining_news_count = len(df)
        duplicates_removed = original_news_count - remaining_news_count
        st.write(f"Из {original_news_count} новостных сообщений удалены {duplicates_removed} дублирующих. Осталось {remaining_news_count}.")

        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize new columns
        df['Translated'] = ''
        df['Sentiment'] = ''
        df['Impact'] = ''
        df['Reasoning'] = ''
        df['Event_Type'] = ''
        df['Event_Summary'] = ''
        
        # Process each news item
        for index, row in df.iterrows():
            try:
                # Translate and analyze sentiment
                translated_text = translator.translate_text(row['Выдержки из текста'])
                df.at[index, 'Translated'] = translated_text
                
                sentiment = analyze_sentiment(translated_text)
                df.at[index, 'Sentiment'] = sentiment
                
                # Detect events
                event_type, event_summary = detect_events(llm, row['Выдержки из текста'], row['Объект'])
                df.at[index, 'Event_Type'] = event_type
                df.at[index, 'Event_Summary'] = event_summary
                
                if sentiment == "Negative":
                    impact, reasoning = estimate_impact(llm, translated_text, row['Объект'])
                    df.at[index, 'Impact'] = impact
                    df.at[index, 'Reasoning'] = reasoning
                
                # Update progress
                progress = (index + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Проанализировано {index + 1} из {len(df)} новостей")
                
            except Exception as e:
                st.warning(f"Ошибка при обработке новости {index + 1}: {str(e)}")
                continue
        
        return df
        
    except Exception as e:
        st.error(f"❌ Ошибка при обработке файла: {str(e)}")
        return df if df is not None else None


def translate_reasoning_to_russian(llm, text):
    template = """
    Translate this English explanation to Russian, maintaining a formal business style:
    "{text}"
    
    Your response should contain only the Russian translation.
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = prompt | llm | RunnablePassthrough()
    response = chain.invoke({"text": text})
    
    # Handle different response types
    if hasattr(response, 'content'):
        return response.content.strip()
    elif isinstance(response, str):
        return response.strip()
    else:
        return str(response).strip()
    

def create_download_section(excel_data, pdf_data):
    st.markdown("""
        <div class="download-container">
            <div class="download-header">📥 Результаты анализа доступны для скачивания:</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        if excel_data is not None:
            st.download_button(
                label="📊 Скачать Excel отчет",
                data=excel_data,
                file_name="результат_анализа.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
        else:
            st.error("Ошибка при создании Excel файла")
    



def display_sentiment_results(row, sentiment, impact=None, reasoning=None):
    if sentiment == "Negative":
        st.markdown(f"""
            <div style='color: red; font-weight: bold;'>
            Объект: {row['Объект']}<br>
            Новость: {row['Заголовок']}<br>
            Тональность: {sentiment}<br>
            {"Эффект: " + impact + "<br>" if impact else ""}
            {"Обоснование: " + reasoning + "<br>" if reasoning else ""}
            </div>
            """, unsafe_allow_html=True)
    elif sentiment == "Positive":
        st.markdown(f"""
            <div style='color: green; font-weight: bold;'>
            Объект: {row['Объект']}<br>
            Новость: {row['Заголовок']}<br>
            Тональность: {sentiment}<br>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write(f"Объект: {row['Объект']}")
        st.write(f"Новость: {row['Заголовок']}")
        st.write(f"Тональность: {sentiment}")
    
    st.write("---")




    
# Initialize sentiment analyzers
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")


def get_mapped_sentiment(result):
    label = result['label'].lower()
    if label in ["positive", "label_2", "pos", "pos_label"]:
        return "Positive"
    elif label in ["negative", "label_0", "neg", "neg_label"]:
        return "Negative"
    return "Neutral"



def analyze_sentiment(text):
    finbert_result = get_mapped_sentiment(finbert(text, truncation=True, max_length=512)[0])
    roberta_result = get_mapped_sentiment(roberta(text, truncation=True, max_length=512)[0])
    finbert_tone_result = get_mapped_sentiment(finbert_tone(text, truncation=True, max_length=512)[0])
    
    # Consider sentiment negative if any model says it's negative
    if any(result == "Negative" for result in [finbert_result, roberta_result, finbert_tone_result]):
        return "Negative"
    elif all(result == "Positive" for result in [finbert_result, roberta_result, finbert_tone_result]):
        return "Positive"
    return "Neutral"

def analyze_sentiment(text):
    finbert_result = get_mapped_sentiment(finbert(text, truncation=True, max_length=512)[0])
    roberta_result = get_mapped_sentiment(roberta(text, truncation=True, max_length=512)[0])
    finbert_tone_result = get_mapped_sentiment(finbert_tone(text, truncation=True, max_length=512)[0])
    
    # Count occurrences of each sentiment
    sentiments = [finbert_result, roberta_result, finbert_tone_result]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    
    # Return sentiment if at least two models agree, otherwise return Neutral
    for sentiment, count in sentiment_counts.items():
        if count >= 2:
            return sentiment
    return "Neutral"


def detect_events(llm, text, entity):
    template = """
    Проанализируйте следующую новость о компании "{entity}" и определите наличие следующих событий:
    1. Публикация отчетности и ключевые показатели (выручка, прибыль, EBITDA)
    2. События на рынке ценных бумаг (погашение облигаций, выплата/невыплата купона, дефолт, реструктуризация)
    3. Судебные иски или юридические действия против компании, акционеров, менеджеров

    Новость: {text}

    Ответьте в следующем формате:
    Тип: ["Отчетность" или "РЦБ" или "Суд" или "Нет"]
    Краткое описание: [краткое описание события на русском языке, не более 2 предложений]
    """
    
    prompt = PromptTemplate(template=template, input_variables=["entity", "text"])
    chain = prompt | llm
    response = chain.invoke({"entity": entity, "text": text})
    
    event_type = "Нет"
    summary = ""
    
    try:
        response_text = response.content if hasattr(response, 'content') else str(response)
        if "Тип:" in response_text and "Краткое описание:" in response_text:
            type_part, summary_part = response_text.split("Краткое описание:")
            event_type = type_part.split("Тип:")[1].strip()
            summary = summary_part.strip()
    except Exception as e:
        st.warning(f"Ошибка при анализе событий: {str(e)}")
    
    return event_type, summary

def fuzzy_deduplicate(df, column, threshold=50):
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


def init_langchain_llm(model_choice):
    try:
        if model_choice == "Groq (llama-3.1-70b)":
            if 'groq_key' not in st.secrets:
                st.error("Groq API key not found in secrets. Please add it with the key 'groq_key'.")
                st.stop()
                
            return ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.1-70b-versatile",
                openai_api_key=st.secrets['groq_key'],
                temperature=0.0
            )
            
        elif model_choice == "ChatGPT-4-mini":
            if 'openai_key' not in st.secrets:
                st.error("OpenAI API key not found in secrets. Please add it with the key 'openai_key'.")
                st.stop()
                
            return ChatOpenAI(
                model="gpt-4",
                openai_api_key=st.secrets['openai_key'],
                temperature=0.0
            )
            
        else:  # Qwen API
            if 'ali_key' not in st.secrets:
                st.error("DashScope API key not found in secrets. Please add it with the key 'dashscope_api_key'.")
                st.stop()
            
            # Using Qwen's API through DashScope
            return ChatOpenAI(
                base_url="https://dashscope.aliyuncs.com/api/v1",
                model="qwen-max",
                openai_api_key=st.secrets['ali_key'],
                temperature=0.0
            )
            
    except Exception as e:
        st.error(f"Error initializing the LLM: {str(e)}")
        st.stop()

def estimate_impact(llm, news_text, entity):
    template = """
    Analyze the following news piece about the entity "{entity}" and estimate its monetary impact in Russian rubles for this entity in the next 6 months.
    
    If precise monetary estimate is not possible, categorize the impact as one of the following:
    1. "Значительный риск убытков" 
    2. "Умеренный риск убытков"
    3. "Незначительный риск убытков"
    4. "Вероятность прибыли"
    5. "Неопределенный эффект"

    Provide brief reasoning (maximum 100 words).

    News: {news}

    Your response should be in the following format:
    Impact: [Your estimate or category]
    Reasoning: [Your reasoning]
    """
    prompt = PromptTemplate(template=template, input_variables=["entity", "news"])
    chain = prompt | llm
    response = chain.invoke({"entity": entity, "news": news_text})
    
    impact = "Неопределенный эффект"
    reasoning = "Не удалось получить обоснование"
    
    # Extract content from response
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    try:
        if "Impact:" in response_text and "Reasoning:" in response_text:
            impact_part, reasoning_part = response_text.split("Reasoning:")
            impact = impact_part.split("Impact:")[1].strip()
            reasoning = reasoning_part.strip()
    except Exception as e:
        st.error(f"Error parsing LLM response: {str(e)}")
    
    return impact, reasoning

def format_elapsed_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} час{'ов' if hours != 1 else ''}")
    if minutes > 0:
        time_parts.append(f"{minutes} минут{'' if minutes == 1 else 'ы' if 2 <= minutes <= 4 else ''}")
    if seconds > 0 or not time_parts:
        time_parts.append(f"{seconds} секунд{'а' if seconds == 1 else 'ы' if 2 <= seconds <= 4 else ''}")
    
    return " ".join(time_parts)

def generate_sentiment_visualization(df):
    negative_df = df[df['Sentiment'] == 'Negative']
    
    if negative_df.empty:
        st.warning("Не обнаружено негативных упоминаний. Отображаем общую статистику по объектам.")
        entity_counts = df['Объект'].value_counts()
    else:
        entity_counts = negative_df['Объект'].value_counts()
    
    if len(entity_counts) == 0:
        st.warning("Нет данных для визуализации.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(entity_counts) * 0.5)))
    entity_counts.plot(kind='barh', ax=ax)
    ax.set_title('Количество негативных упоминаний по объектам')
    ax.set_xlabel('Количество упоминаний')
    plt.tight_layout()
    return fig

def create_analysis_data(df):
    analysis_data = []
    for _, row in df.iterrows():
        if row['Sentiment'] == 'Negative':
            analysis_data.append([
                row['Объект'], 
                row['Заголовок'], 
                'РИСК УБЫТКА', 
                row['Impact'],
                row['Reasoning'],
                row['Выдержки из текста']
            ])
    return pd.DataFrame(analysis_data, columns=[
        'Объект', 
        'Заголовок', 
        'Признак', 
        'Оценка влияния',
        'Обоснование',
        'Текст сообщения'
    ])

def create_output_file(df, uploaded_file, llm):
    wb = load_workbook("sample_file.xlsx")
    
    try:
        # Update 'Мониторинг' sheet with events
        ws = wb['Мониторинг']
        row_idx = 4
        for _, row in df.iterrows():
            if row['Event_Type'] != 'Нет':
                ws.cell(row=row_idx, column=5, value=row['Объект'])  # Column E
                ws.cell(row=row_idx, column=6, value=row['Заголовок'])  # Column F
                ws.cell(row=row_idx, column=7, value=row['Event_Type'])  # Column G
                ws.cell(row=row_idx, column=8, value=row['Event_Summary'])  # Column H
                ws.cell(row=row_idx, column=9, value=row['Выдержки из текста'])  # Column I
                row_idx += 1
                   
        # Sort entities by number of negative publications
        entity_stats = pd.DataFrame({
            'Объект': df['Объект'].unique(),
            'Всего': df.groupby('Объект').size(),
            'Негативные': df[df['Sentiment'] == 'Negative'].groupby('Объект').size().fillna(0).astype(int),
            'Позитивные': df[df['Sentiment'] == 'Positive'].groupby('Объект').size().fillna(0).astype(int)
        }).sort_values('Негативные', ascending=False)
        
        # Calculate most negative impact for each entity
        entity_impacts = {}
        for entity in df['Объект'].unique():
            entity_df = df[df['Объект'] == entity]
            negative_impacts = entity_df[entity_df['Sentiment'] == 'Negative']['Impact']
            entity_impacts[entity] = negative_impacts.iloc[0] if len(negative_impacts) > 0 else 'Неопределенный эффект'
        
        # Update 'Сводка' sheet
        ws = wb['Сводка']
        for idx, (entity, row) in enumerate(entity_stats.iterrows(), start=4):
            ws.cell(row=idx, column=5, value=entity)  # Column E
            ws.cell(row=idx, column=6, value=row['Всего'])  # Column F
            ws.cell(row=idx, column=7, value=row['Негативные'])  # Column G
            ws.cell(row=idx, column=8, value=row['Позитивные'])  # Column H
            ws.cell(row=idx, column=9, value=entity_impacts[entity])  # Column I
        
        # Update 'Значимые' sheet
        ws = wb['Значимые']
        row_idx = 3
        for _, row in df.iterrows():
            if row['Sentiment'] in ['Negative', 'Positive']:
                ws.cell(row=row_idx, column=3, value=row['Объект'])  # Column C
                ws.cell(row=row_idx, column=4, value='релевантно')   # Column D
                ws.cell(row=row_idx, column=5, value=row['Sentiment']) # Column E
                ws.cell(row=row_idx, column=6, value=row['Impact'])   # Column F
                ws.cell(row=row_idx, column=7, value=row['Заголовок']) # Column G
                ws.cell(row=row_idx, column=8, value=row['Выдержки из текста']) # Column H
                row_idx += 1
        
        # Copy 'Публикации' sheet
        original_df = pd.read_excel(uploaded_file, sheet_name='Публикации')
        ws = wb['Публикации']
        for r_idx, row in enumerate(dataframe_to_rows(original_df, index=False, header=True), start=1):
            for c_idx, value in enumerate(row, start=1):
                ws.cell(row=r_idx, column=c_idx, value=value)
        
        # Update 'Анализ' sheet
        ws = wb['Анализ']
        row_idx = 4
        for _, row in df[df['Sentiment'] == 'Negative'].iterrows():
            ws.cell(row=row_idx, column=5, value=row['Объект'])  # Column E
            ws.cell(row=row_idx, column=6, value=row['Заголовок'])  # Column F
            ws.cell(row=row_idx, column=7, value="Риск убытка")  # Column G
            
            # Translate reasoning if it exists
            if pd.notna(row['Reasoning']):
                translated_reasoning = translate_reasoning_to_russian(llm, row['Reasoning'])
                ws.cell(row=row_idx, column=8, value=translated_reasoning)  # Column H
            
            ws.cell(row=row_idx, column=9, value=row['Выдержки из текста'])  # Column I
            row_idx += 1
        
        # Update 'Тех.приложение' sheet
        tech_df = df[['Объект', 'Заголовок', 'Выдержки из текста', 'Translated', 'Sentiment', 'Impact', 'Reasoning']]
        if 'Тех.приложение' not in wb.sheetnames:
            wb.create_sheet('Тех.приложение')
        ws = wb['Тех.приложение']
        for r_idx, row in enumerate(dataframe_to_rows(tech_df, index=False, header=True), start=1):
            for c_idx, value in enumerate(row, start=1):
                ws.cell(row=r_idx, column=c_idx, value=value)
    
    except Exception as e:
        st.warning(f"Ошибка при создании выходного файла: {str(e)}")
    
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output

def main():
    with st.sidebar:
        st.title("::: AI-анализ мониторинга новостей (v.3.33 ):::")
        st.subheader("по материалам СКАН-ИНТЕРФАКС ")
        
        model_choice = st.radio(
            "Выберите модель для анализа:",
            ["Groq (llama-3.1-70b)", "ChatGPT-4-mini", "Qwen-Max"],
            key="model_selector"
        )
        
        translation_method = st.radio(
            "Выберите метод перевода:",
            ["googletrans", "llm"],
            key="translation_selector",
            help="googletrans - быстрее, llm - качественнее, но медленнее"
        )
        
        st.markdown(
        """
        Использованы технологии:  
        - Анализ естественного языка с помощью предтренированных нейросетей **BERT**,<br/>
	    - Дополнительная обработка при помощи больших языковых моделей (**LLM**),<br/>
	    - объединенные при помощи	фреймворка **LangChain**.<br>
        """,
        unsafe_allow_html=True)

        with st.expander("ℹ️ Инструкция"):
            st.markdown("""
            1. Выберите модель для анализа
            2. Выберите метод перевода
            3. Загрузите Excel файл с новостями
            4. Дождитесь завершения анализа
            5. Скачайте результаты анализа в формате Excel
            """, unsafe_allow_html=True)

   
        st.markdown(
        """
        <style>
        .signature {
            position: fixed;
            right: 12px;
            up: 12px;
            font-size: 14px;
            color: #FF0000;
            opacity: 0.9;
            z-index: 999;
        }
        </style>
        <div class="signature">denis.pokrovsky.npff</div>
        """,
        unsafe_allow_html=True
        )

    st.title("Анализ мониторинга новостей")
    
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    
    # Single file uploader with unique key
    uploaded_file = st.sidebar.file_uploader("Выбирайте Excel-файл", type="xlsx", key="unique_file_uploader")
    
    if uploaded_file is not None and st.session_state.processed_df is None:
        start_time = time.time()
        
        # Initialize LLM with selected model
        llm = init_langchain_llm(model_choice)

        # Process file with selected translation method
        st.session_state.processed_df = process_file(
            uploaded_file, 
            model_choice,
            translation_method
        )

        st.subheader("Предпросмотр данных")
        preview_df = st.session_state.processed_df[['Объект', 'Заголовок', 'Sentiment', 'Impact']].head()
        st.dataframe(preview_df)
        
        # Add preview of Monitoring results
        st.subheader("Предпросмотр мониторинга событий и риск-факторов эмитентов")
        monitoring_df = st.session_state.processed_df[
            (st.session_state.processed_df['Event_Type'] != 'Нет') & 
            (st.session_state.processed_df['Event_Type'].notna())
        ][['Объект', 'Заголовок', 'Event_Type', 'Event_Summary']].head()
        
        if len(monitoring_df) > 0:
            st.dataframe(monitoring_df)
        else:
            st.info("Не обнаружено значимых событий для мониторинга")


        analysis_df = create_analysis_data(st.session_state.processed_df)
        st.subheader("Анализ")
        st.dataframe(analysis_df)
        
       
        output = create_output_file(st.session_state.processed_df, uploaded_file, llm)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_elapsed_time(elapsed_time)
        st.success(f"Обработка и анализ завершены за {formatted_time}.")

        st.download_button(
            label="Скачать результат анализа",
            data=output,
            file_name="результат_анализа.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()