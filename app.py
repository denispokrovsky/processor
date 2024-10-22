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


def create_download_section(excel_data, output_capture):
    st.markdown("""
        <style>
        .download-container {
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin: 20px 0;
        }
        .download-header {
            color: #0066cc;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        <div class="download-container">
            <div class="download-header">📥 Результаты анализа готовы к скачиванию:</div>
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
    
    with col2:
        try:
            pdf_data = generate_pdf_report(output_capture.texts)
            if pdf_data:
                st.download_button(
                    label="📄 Скачать протокол",
                    data=pdf_data,
                    file_name="протокол_анализа.pdf" if isinstance(pdf_data, bytes) else "протокол_анализа.txt",
                    mime="application/pdf" if isinstance(pdf_data, bytes) else "text/plain",
                    key="pdf_download"
                )
            else:
                st.error("Ошибка при создании протокола")
        except Exception as e:
            st.error(f"Ошибка при создании протокола: {str(e)}")

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


class StreamlitCapture:
    def __init__(self):
        self.texts = []
    
    def write(self, text):
        if text and str(text).strip():  # Only capture non-empty text
            self.texts.append(str(text))
    
    def flush(self):
        pass



def generate_pdf_report(texts):
    try:
        import pdfkit
        from jinja2 import Template
        
        html_content = """
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; }
                .content { margin: 20px; }
            </style>
        </head>
        <body>
            <div class="content">
                {% for text in texts %}
                    <p>{{ text }}</p>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        
        template = Template(html_content)
        rendered_html = template.render(texts=texts)
        
        # Create PDF in memory
        pdf_data = pdfkit.from_string(rendered_html, False)
        return pdf_data
        
    except Exception as e:
        st.warning(f"Не удалось создать PDF отчет: {str(e)}")
        # Return the text as bytes if PDF generation fails
        return '\n'.join(texts).encode('utf-8')
    
# Initialize sentiment analyzers
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")


def translate_text(llm, text):
    template = """
    Translate this Russian text into English:
    "{text}"
    
    Your response should contain only the English translation.
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = prompt | llm | RunnablePassthrough()
    response = chain.invoke({"text": text})
    
    # Handle different response types
    if hasattr(response, 'content'):  # If it's an AIMessage object
        return response.content.strip()
    elif isinstance(response, str):    # If it's a string
        return response.strip()
    else:
        return str(response).strip()   # Convert any other type to string

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

def init_langchain_llm():
    try:
        if 'groq_key' in st.secrets:
            groq_api_key = st.secrets['groq_key']
        else:
            st.error("Groq API key not found in Hugging Face secrets. Please add it with the key 'groq_key'.")
            st.stop()

        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-70b-versatile",
            openai_api_key=groq_api_key,  # Updated parameter name
            temperature=0.0
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing the Groq LLM: {str(e)}")
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

def process_file(uploaded_file):
    output_capture = StreamlitCapture()
    old_stdout = sys.stdout
    sys.stdout = output_capture
    
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Публикации')

        required_columns = ['Объект', 'Заголовок', 'Выдержки из текста']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Error: The following required columns are missing from the input file: {', '.join(missing_columns)}")
            st.stop()
    
        # Initialize LLM
        llm = init_langchain_llm()
        if not llm:
            st.error("Не удалось инициализировать нейросеть. Пожалуйста, проверьте настройки и попробуйте снова.")
            st.stop()

        # Deduplication
        original_news_count = len(df)
        df = df.groupby('Объект', group_keys=False).apply(
            lambda x: fuzzy_deduplicate(x, 'Выдержки из текста', 65)
        ).reset_index(drop=True)
    
        remaining_news_count = len(df)
        duplicates_removed = original_news_count - remaining_news_count
        st.write(f"Из {original_news_count} новостных сообщений удалены {duplicates_removed} дублирующих. Осталось {remaining_news_count}.")

        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
    
        # Process each news item
        df['Translated'] = ''
        df['Sentiment'] = ''
        df['Impact'] = ''
        df['Reasoning'] = ''
    
        for index, row in df.iterrows():
            translated_text = translate_text(llm, row['Выдержки из текста'])
            df.at[index, 'Translated'] = translated_text
            
            sentiment = analyze_sentiment(translated_text)
            df.at[index, 'Sentiment'] = sentiment
            
            if sentiment == "Negative":
                impact, reasoning = estimate_impact(llm, translated_text, row['Объект'])
                df.at[index, 'Impact'] = impact
                df.at[index, 'Reasoning'] = reasoning
            
            # Update progress
            progress = (index + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Проанализировано {index + 1} из {len(df)} новостей")
            
            # Display results with color coding
            display_sentiment_results(row, sentiment, 
                                   impact if sentiment == "Negative" else None,
                                   reasoning if sentiment == "Negative" else None)
        
       
        sys.stdout = old_stdout
        
        # Prepare Excel file
        excel_output = create_output_file(df, uploaded_file)
        
        # Show success message
        st.success(f"✅ Обработка и анализ завершены за умеренное время.")
        
        # Create download section
        create_download_section(excel_output, output_capture)
        
        return df
        
    except Exception as e:
        sys.stdout = old_stdout
        st.error(f"❌ Ошибка при обработке файла: {str(e)}")
        raise e

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

def create_output_file(df, uploaded_file):
    wb = load_workbook("sample_file.xlsx")
    
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
    
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output

def main():
    
    with st.expander("ℹ️ Как пользоваться"):
        st.markdown("""
        1. Загрузите Excel файл с новостями
        2. Дождитесь завершения анализа
        3. Скачайте результаты анализа в нужном формате (Excel и/или PDF)
        """)    
    
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
    
    st.title("::: анализ мониторинга новостей СКАН-ИНТЕРФАКС (v.3.62):::")
    
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    
    uploaded_file = st.file_uploader("Выбирайте Excel-файл", type="xlsx")
    
    if uploaded_file is not None and st.session_state.processed_df is None:
        start_time = time.time()
        
        st.session_state.processed_df = process_file(uploaded_file)

       
        st.subheader("Предпросмотр данных")
        preview_df = st.session_state.processed_df[['Объект', 'Заголовок', 'Sentiment', 'Impact']].head()
        st.dataframe(preview_df)
        
        analysis_df = create_analysis_data(st.session_state.processed_df)
        st.subheader("Анализ")
        st.dataframe(analysis_df)
        
        output = create_output_file(st.session_state.processed_df, uploaded_file)
        
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