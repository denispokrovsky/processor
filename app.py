import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import io
from rapidfuzz import fuzz
import os
from openpyxl import load_workbook
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
            api_key=groq_api_key,
            temperature=0.0
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing the Groq LLM: {str(e)}")
        st.stop()

def estimate_sentiment_and_impact(llm, news_text, entity):
    template = """
    Проанализируйте следующую новость об объекте "{entity}" и определите:
    1. Тональность новости (Позитивная/Негативная/Нейтральная)
    2. Оцените потенциальное финансовое влияние в рублях для этого объекта в ближайшие 6 месяцев.
    
    Если точную денежную оценку дать невозможно, категоризируйте влияние как одно из следующих:
    1. "Значительный риск убытков" 
    2. "Умеренный риск убытков"
    3. "Незначительный риск убытков"
    4. "Вероятность прибыли"
    5. "Неопределенный эффект"

    Также предоставьте краткое обоснование (максимум 100 слов).

    Новость: {news}

    Ответ дайте в следующем формате:
    Sentiment: [Positive/Negative/Neutral]
    Impact: [Ваша оценка или категория]
    Reasoning: [Ваше обоснование]
    """
    prompt = PromptTemplate(template=template, input_variables=["entity", "news"])
    chain = prompt | llm | RunnablePassthrough()
    response = chain.invoke({"entity": entity, "news": news_text})
    
    sentiment = "Neutral"
    impact = "Неопределенный эффект"
    reasoning = "Не удалось получить обоснование"
    
    if isinstance(response, str):
        try:
            if "Sentiment:" in response:
                sentiment_part = response.split("Sentiment:")[1].split("\n")[0].strip().lower()
                if "positive" in sentiment_part:
                    sentiment = "Positive"
                elif "negative" in sentiment_part:
                    sentiment = "Negative"
            
            if "Impact:" in response and "Reasoning:" in response:
                impact_part, reasoning_part = response.split("Reasoning:")
                impact = impact_part.split("Impact:")[1].strip()
                reasoning = reasoning_part.strip()
        except Exception as e:
            st.error(f"Error parsing LLM response: {str(e)}")
    
    return sentiment, impact, reasoning

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
    df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    
    required_columns = ['Объект', 'Заголовок', 'Выдержки из текста']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Error: The following required columns are missing from the input file: {', '.join(missing_columns)}")
        st.stop()
    
    original_news_count = len(df)
    df = df.groupby('Объект').apply(
        lambda x: fuzzy_deduplicate(x, 'Выдержки из текста', 65)
    ).reset_index(drop=True)

    remaining_news_count = len(df)
    duplicates_removed = original_news_count - remaining_news_count
    st.write(f"Из {original_news_count} новостных сообщений удалены {duplicates_removed} дублирующих. Осталось {remaining_news_count}.")

    llm = init_langchain_llm()
    if not llm:
        st.error("Не удалось инициализировать нейросеть. Пожалуйста, проверьте настройки и попробуйте снова.")
        st.stop()

    df['Sentiment'] = ''
    df['Impact'] = ''
    df['Reasoning'] = ''
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, row in df.iterrows():
        sentiment, impact, reasoning = estimate_sentiment_and_impact(
            llm, 
            row['Выдержки из текста'], 
            row['Объект']
        )
        
        df.at[index, 'Sentiment'] = sentiment
        df.at[index, 'Impact'] = impact
        df.at[index, 'Reasoning'] = reasoning
        
        progress = (index + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Проанализировано {index + 1} из {len(df)} новостей")
        
        st.write(f"Объект: {row['Объект']}")
        st.write(f"Новость: {row['Заголовок']}")
        st.write(f"Тональность: {sentiment}")
        st.write(f"Эффект: {impact}")
        st.write(f"Обоснование: {reasoning}")
        st.write("---")

    progress_bar.empty()
    status_text.empty()

    visualization = generate_sentiment_visualization(df)
    if visualization:
        st.pyplot(visualization)

    return df

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
    
    summary_df = pd.DataFrame({
        'Объект': df['Объект'].unique(),
        'Всего новостей': df.groupby('Объект').size(),
        'Негативные': df[df['Sentiment'] == 'Negative'].groupby('Объект').size().fillna(0).astype(int),
        'Позитивные': df[df['Sentiment'] == 'Positive'].groupby('Объект').size().fillna(0).astype(int),
        'Преобладающий эффект': df.groupby('Объект')['Impact'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'Неопределенный'
        )
    })
    
    summary_df = summary_df.sort_values('Негативные', ascending=False)
    
    ws = wb['Сводка']
    for r_idx, row in enumerate(dataframe_to_rows(summary_df, index=False, header=True), start=4):
        for c_idx, value in enumerate(row, start=5):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    significant_data = []
    for _, row in df.iterrows():
        if row['Sentiment'] in ['Negative', 'Positive']:
            significant_data.append([
                row['Объект'],
                'релевантен',
                row['Sentiment'],
                row['Impact'],
                row['Заголовок'],
                row['Выдержки из текста']
            ])
    
    ws = wb['Значимые']
    for r_idx, row in enumerate(significant_data, start=3):
        for c_idx, value in enumerate(row, start=3):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    analysis_df = create_analysis_data(df)
    ws = wb['Анализ']
    for r_idx, row in enumerate(dataframe_to_rows(analysis_df, index=False, header=True), start=4):
        for c_idx, value in enumerate(row, start=5):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    original_df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    ws = wb['Публикации']
    for r_idx, row in enumerate(dataframe_to_rows(original_df, index=False, header=True), start=1):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
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

def main():
    st.markdown(
        """
        <style>
        .signature {
            position: fixed;
            right: 12px;
            bottom: 12px;
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
    
    st.title("... приступим к анализу... версия 75")
    
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