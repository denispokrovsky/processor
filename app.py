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
from io import StringIO, BytesIO
import sys
import contextlib
from langchain_openai import ChatOpenAI  # Updated import
import pdfkit
from jinja2 import Template
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional
import torch
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM  # 4 Qwen
)

from threading import Event
import threading
from queue import Queue

from deep_translator import GoogleTranslator
from googletrans import Translator as LegacyTranslator
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px


class ProcessControl:
    def __init__(self):
        self.pause_event = Event()
        self.stop_event = Event()
        self.pause_event.set()  # Start in non-paused state
        
    def pause(self):
        self.pause_event.clear()
        
    def resume(self):
        self.pause_event.set()
        
    def stop(self):
        self.stop_event.set()
        self.pause_event.set()  # Ensure not stuck in pause
        
    def reset(self):
        self.stop_event.clear()
        self.pause_event.set()
        
    def is_paused(self):
        return not self.pause_event.is_set()
        
    def is_stopped(self):
        return self.stop_event.is_set()
        
    def wait_if_paused(self):
        self.pause_event.wait()


class FallbackLLMSystem:
    def __init__(self):
        """Initialize fallback models for event detection and reasoning"""
        try:
            # Initialize MT5 model (multilingual T5)
            self.model_name = "google/mt5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            st.success(f"пока все в порядке: запущена MT5 model на = {self.device} =")
            
        except Exception as e:
            st.error(f"Ошибка запуска модели MT5: {str(e)}")
            raise

    def invoke(self, prompt_args):
        """Make the class compatible with LangChain by implementing invoke"""
        try:
            if isinstance(prompt_args, dict):
                # Extract the prompt template result
                template_result = prompt_args.get('template_result', '')
                if not template_result:
                    # Try to construct from entity and news if available
                    entity = prompt_args.get('entity', '')
                    news = prompt_args.get('news', '')
                    template_result = f"Analyze news about {entity}: {news}"
            else:
                template_result = str(prompt_args)

            # Process with MT5
            inputs = self.tokenizer(
                template_result,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return in a format compatible with LangChain
            return type('Response', (), {'content': response})()
            
        except Exception as e:
            st.warning(f"MT5 generation error: {str(e)}")
            # Return a default response on error
            return type('Response', (), {
                'content': 'Impact: Неопределенный эффект\nReasoning: Ошибка анализа'
            })()

    def __or__(self, other):
        """Implement the | operator for chain compatibility"""
        if callable(other):
            return lambda x: other(self(x))
        return NotImplemented

    def __rrshift__(self, other):
        """Implement the >> operator for chain compatibility"""
        return self.__or__(other)

    def __call__(self, prompt_args):
        """Make the class callable for chain compatibility"""
        return self.invoke(prompt_args)

    def detect_events(self, text: str, entity: str) -> tuple[str, str]:
        """
        Detect events using MT5 with improved error handling and response parsing
        
        Args:
            text (str): The news text to analyze
            entity (str): The company/entity name
            
        Returns:
            tuple[str, str]: (event_type, summary)
        """
        # Initialize default return values
        event_type = "Нет"
        summary = ""
        
        # Input validation
        if not text or not entity or not isinstance(text, str) or not isinstance(entity, str):
            return event_type, "Invalid input"
            
        try:
            # Clean and prepare input text
            text = text.strip()
            entity = entity.strip()
            
            # Construct prompt with better formatting
            prompt = f"""<s>Analyze the following news about {entity}:

    Text: {text}

    Task: Identify the main event type and provide a brief summary.

    Event types:
    1. Отчетность - Events related to financial reports, earnings, revenue, EBITDA
    2. РЦБ - Events related to securities, bonds, stock market, defaults, restructuring
    3. Суд - Events related to legal proceedings, lawsuits, arbitration
    4. Нет - No significant events detected

    Required output format:
    Тип: [event type]
    Краткое описание: [1-2 sentence summary]</s>"""

            # Process with MT5
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=300,  # Increased for better summaries
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3  # Prevent repetition
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            except torch.cuda.OutOfMemoryError:
                st.warning("GPU memory exceeded, falling back to CPU")
                self.model = self.model.to('cpu')
                inputs = inputs.to('cpu')
                outputs = self.model.generate(
                    **inputs,
                    max_length=300,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                self.model = self.model.to(self.device)  # Move back to GPU
                
            # Enhanced response parsing
            if "Тип:" in response and "Краткое описание:" in response:
                try:
                    # Split and clean parts
                    parts = response.split("Краткое описание:")
                    type_part = parts[0].split("Тип:")[1].strip()
                    
                    # Validate event type with fuzzy matching
                    valid_types = ["Отчетность", "РЦБ", "Суд", "Нет"]
                    
                    # Check for exact matches first
                    if type_part in valid_types:
                        event_type = type_part
                    else:
                        # Check keywords for each type
                        keywords = {
                            "Отчетность": ["отчет", "выручка", "прибыль", "ebitda", "финанс"],
                            "РЦБ": ["облигаци", "купон", "дефолт", "реструктуризац", "ценные бумаги"],
                            "Суд": ["суд", "иск", "арбитраж", "разбирательств"]
                        }
                        
                        # Look for keywords in both type and summary
                        full_text = response.lower()
                        for event_category, category_keywords in keywords.items():
                            if any(keyword in full_text for keyword in category_keywords):
                                event_type = event_category
                                break
                    
                    # Extract and clean summary
                    if len(parts) > 1:
                        summary = parts[1].strip()
                        # Ensure summary isn't too long
                        if len(summary) > 200:
                            summary = summary[:197] + "..."
                        
                        # Add entity reference if missing
                        if entity.lower() not in summary.lower():
                            summary = f"Компания {entity}: {summary}"
                    
                except IndexError:
                    st.warning("Error parsing model response format")
                    return "Нет", "Error parsing response"
                    
            # Additional validation
            if not summary or len(summary) < 5:
                keywords = {
                    "Отчетность": "Обнаружена информация о финансовой отчетности",
                    "РЦБ": "Обнаружена информация о ценных бумагах",
                    "Суд": "Обнаружена информация о судебном разбирательстве",
                    "Нет": "Значимых событий не обнаружено"
                }
                summary = f"{keywords.get(event_type, 'Требуется дополнительный анализ')} ({entity})"
                
            return event_type, summary
                
        except Exception as e:
            st.warning(f"Event detection error: {str(e)}")
            # Try to provide more specific error information
            if "CUDA" in str(e):
                return "Нет", "GPU error - falling back to CPU needed"
            elif "tokenizer" in str(e):
                return "Нет", "Text processing error"
            elif "model" in str(e):
                return "Нет", "Model inference error"
            else:
                return "Нет", "Ошибка анализа"
        

def ensure_groq_llm():
    """Initialize Groq LLM for impact estimation"""
    try:
        if 'groq_key' not in st.secrets:
            st.error("Groq API key not found in secrets. Please add it with the key 'groq_key'.")
            return None
            
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-70b-versatile",
            openai_api_key=st.secrets['groq_key'],
            temperature=0.0
        )
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        return None

def estimate_impact(llm, news_text, entity):
    """
    Estimate impact using Groq LLM regardless of the main model choice.
    Falls back to the provided LLM if Groq initialization fails.
    """
    # Initialize default return values
    impact = "Неопределенный эффект"
    reasoning = "Не удалось получить обоснование"
    
    try:
        # Always try to use Groq first
        groq_llm = ensure_groq_llm()
        working_llm = groq_llm if groq_llm is not None else llm
        
        template = """
        You are a financial analyst. Analyze this news piece about {entity} and assess its potential impact.
        
        News: {news}
        
        Classify the impact into one of these categories:
        1. "Значительный риск убытков" (Significant loss risk)
        2. "Умеренный риск убытков" (Moderate loss risk)
        3. "Незначительный риск убытков" (Minor loss risk)
        4. "Вероятность прибыли" (Potential profit)
        5. "Неопределенный эффект" (Uncertain effect)
        
        Provide a brief, fact-based reasoning for your assessment.
        
        Format your response exactly as:
        Impact: [category]
        Reasoning: [explanation in 2-3 sentences]
        """
        
        prompt = PromptTemplate(template=template, input_variables=["entity", "news"])
        chain = prompt | working_llm
        response = chain.invoke({"entity": entity, "news": news_text})
        
        # Extract content from response
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        if "Impact:" in response_text and "Reasoning:" in response_text:
            impact_part, reasoning_part = response_text.split("Reasoning:")
            impact_temp = impact_part.split("Impact:")[1].strip()
            
            # Validate impact category
            valid_impacts = [
                "Значительный риск убытков",
                "Умеренный риск убытков",
                "Незначительный риск убытков",
                "Вероятность прибыли",
                "Неопределенный эффект"
            ]
            if impact_temp in valid_impacts:
                impact = impact_temp
            reasoning = reasoning_part.strip()
            
    except Exception as e:
        st.warning(f"Error in impact estimation: {str(e)}")
    
    return impact, reasoning
   
class QwenSystem:
    def __init__(self):
        """Initialize Qwen 2.5 Coder model"""
        try:
            self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
            
            # Initialize model with auto settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            st.success(f"запустил Qwen2.5 model")
            
        except Exception as e:
            st.error(f"ошибка запуска Qwen2.5: {str(e)}")
            raise

    def invoke(self, messages):
        """Process messages using Qwen's chat template"""
        try:
            # Prepare messages with system prompt
            chat_messages = [
                {"role": "system", "content": "You are wise financial analyst. You are a helpful assistant."}
            ]
            chat_messages.extend(messages)
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare model inputs
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract new tokens
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode response
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Return in ChatOpenAI-compatible format
            return type('Response', (), {'content': response})()
            
        except Exception as e:
            st.warning(f"Qwen generation error: {str(e)}")
            raise


class ProcessingUI:
    def __init__(self):
        if 'control' not in st.session_state:
            st.session_state.control = ProcessControl()
            
        # Initialize processing stats in session state if not exists
        if 'processing_stats' not in st.session_state:
            st.session_state.processing_stats = {
                'start_time': time.time(),
                'entities': {},
                'events_timeline': [],
                'negative_alerts': [],
                'processing_speed': []
            }
        
        # Create main layout
        self.setup_layout()
        
    def setup_layout(self):
        """Setup the main UI layout with tabs and sections"""
        # Control Panel
        with st.container():
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                if st.button(
                    "⏸️ Пауза" if not st.session_state.control.is_paused() else "▶️ Продолжить",
                    use_container_width=True
                ):
                    if st.session_state.control.is_paused():
                        st.session_state.control.resume()
                    else:
                        st.session_state.control.pause()
            with col2:
                if st.button("⏹️ Остановить", use_container_width=True):
                    st.session_state.control.stop()
            with col3:
                self.timer_display = st.empty()
        
        # Progress Bar with custom styling
        st.markdown("""
            <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #FF6B6B, #4ECDC4);
            }
            </style>""", 
            unsafe_allow_html=True
        )
        self.progress_bar = st.progress(0)
        self.status = st.empty()

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Основные метрики", 
            "🏢 По организациям", 
            "⚠️ Важные события", 
            "📈 Аналитика"
        ])
        
        with tab1:
            self.setup_main_metrics_tab()
            
        with tab2:
            self.setup_entity_tab()
            
        with tab3:
            self.setup_events_tab()
            
        with tab4:
            self.setup_analytics_tab()
            
    def setup_main_metrics_tab(self):
        """Setup the main metrics display"""
        # Create metrics containers
        metrics_cols = st.columns(4)
        self.total_processed = metrics_cols[0].empty()
        self.negative_count = metrics_cols[1].empty()
        self.events_count = metrics_cols[2].empty()
        self.speed_metric = metrics_cols[3].empty()
        
        # Recent items container with custom styling
        st.markdown("""
            <style>
            .recent-item {
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                background-color: #f0f2f6;
            }
            .negative-item {
                border-left: 4px solid #FF6B6B;
            }
            .positive-item {
                border-left: 4px solid #4ECDC4;
            }
            .event-item {
                border-left: 4px solid #FFE66D;
            }
            </style>
        """, unsafe_allow_html=True)
        
        self.recent_items_container = st.empty()
        
    def setup_entity_tab(self):
        """Setup the entity-wise analysis display"""
        # Entity filter
        self.entity_filter = st.multiselect(
            "Фильтр по организациям:",
            options=[],  # Will be populated as entities are processed
            default=None
        )
        
        # Entity metrics
        self.entity_cols = st.columns([2,1,1,1])
        self.entity_chart = st.empty()
        self.entity_table = st.empty()
        
    def setup_events_tab(self):
        """Setup the events timeline display"""
        # Event type filter
        self.event_filter = st.multiselect(
            "Тип события:",
            options=["Отчетность", "РЦБ", "Суд"],
            default=None
        )
        
        # Timeline container
        self.timeline_container = st.container()
        
    def setup_analytics_tab(self):
        """Setup the analytics display"""
        # Processing speed chart
        self.speed_chart = st.empty()
        
        # Sentiment distribution pie chart
        self.sentiment_chart = st.empty()
        
        # Entity correlation matrix
        self.correlation_chart = st.empty()
        
    def update_stats(self, row, sentiment, event_type, processing_speed):
        """Update all statistics and displays"""
        # Update session state stats
        stats = st.session_state.processing_stats
        entity = row['Объект']
        
        # Update entity stats
        if entity not in stats['entities']:
            stats['entities'][entity] = {
                'total': 0,
                'negative': 0,
                'events': 0,
                'timeline': []
            }
        
        stats['entities'][entity]['total'] += 1
        if sentiment == 'Negative':
            stats['entities'][entity]['negative'] += 1
        if event_type != 'Нет':
            stats['entities'][entity]['events'] += 1
            
        # Update processing speed
        stats['processing_speed'].append(processing_speed)
        
        # Update UI components
        self._update_main_metrics(row, sentiment, event_type, processing_speed)
        self._update_entity_view()
        self._update_events_view(row, event_type)
        self._update_analytics()
        
    def _update_main_metrics(self, row, sentiment, event_type, speed):
        """Update main metrics tab"""
        total = sum(e['total'] for e in st.session_state.processing_stats['entities'].values())
        total_negative = sum(e['negative'] for e in st.session_state.processing_stats['entities'].values())
        total_events = sum(e['events'] for e in st.session_state.processing_stats['entities'].values())
        
        # Update metrics
        self.total_processed.metric("Обработано", total)
        self.negative_count.metric("Негативных", total_negative)
        self.events_count.metric("Событий", total_events)
        self.speed_metric.metric("Скорость", f"{speed:.1f} сообщ/сек")
        
        # Update recent items
        self._update_recent_items(row, sentiment, event_type)
        
    def _update_recent_items(self, row, sentiment, event_type):
        """Update recent items display with custom styling"""
        items_html = "<div style='max-height: 300px; overflow-y: auto;'>"
        
        # Add new item to the beginning of the list
        item_class = 'recent-item '
        if sentiment == 'Negative':
            item_class += 'negative-item'
        elif sentiment == 'Positive':
            item_class += 'positive-item'
        if event_type != 'Нет':
            item_class += ' event-item'
            
        items_html += f"""
            <div class='{item_class}'>
                <strong>{row['Объект']}</strong><br>
                {row['Заголовок']}<br>
                <small>
                    Тональность: {sentiment} 
                    {f" | Событие: {event_type}" if event_type != 'Нет' else ""}
                </small>
            </div>
        """
        
        items_html += "</div>"
        self.recent_items_container.markdown(items_html, unsafe_allow_html=True)
        
    def _update_entity_view(self):
        """Update entity tab visualizations"""
        stats = st.session_state.processing_stats['entities']
        if not stats:
            return
            
        # Update entity filter options
        current_options = set(self.entity_filter)
        all_entities = set(stats.keys())
        if current_options != all_entities:
            self.entity_filter = list(all_entities)
            
        # Create entity comparison chart using Plotly
        df_entities = pd.DataFrame.from_dict(stats, orient='index')
        fig = go.Figure(data=[
            go.Bar(name='Total', x=df_entities.index, y=df_entities['total']),
            go.Bar(name='Negative', x=df_entities.index, y=df_entities['negative']),
            go.Bar(name='Events', x=df_entities.index, y=df_entities['events'])
        ])
        fig.update_layout(barmode='group', title='Entity Statistics')
        self.entity_chart.plotly_chart(fig, use_container_width=True)
        
    def _update_events_view(self, row, event_type):
        """Update events timeline"""
        if event_type != 'Нет':
            event_html = f"""
                <div class='timeline-item'>
                    <div class='timeline-marker'></div>
                    <div class='timeline-content'>
                        <h3>{event_type}</h3>
                        <p><strong>{row['Объект']}</strong></p>
                        <p>{row['Заголовок']}</p>
                        <small>{datetime.now().strftime('%H:%M:%S')}</small>
                    </div>
                </div>
            """
            with self.timeline_container:
                st.markdown(event_html, unsafe_allow_html=True)
                
    def _update_analytics(self):
        """Update analytics tab visualizations"""
        stats = st.session_state.processing_stats
        
        # Processing speed chart
        speeds = stats['processing_speed'][-1:]  # Last 1 measurement
        fig_speed = go.Figure(data=go.Scatter(y=speeds, mode='lines'))
        fig_speed.update_layout(title='Processing Speed Over Time')
        self.speed_chart.plotly_chart(fig_speed, use_container_width=True)
        
        # Add other analytics visualizations as needed
        
    def update_progress(self, current, total):
        """Update progress bar and elapsed time"""
        progress = current / total
        self.progress_bar.progress(progress)
        self.status.text(f"Обрабатываем {current} из {total} сообщений...")
        
        # Update elapsed time
        elapsed = time.time() - st.session_state.processing_stats['start_time']
        self.timer_display.markdown(f"⏱️ {format_elapsed_time(elapsed)}")


class EventDetectionSystem:
    def __init__(self):
        try:
            # Initialize models with specific labels
            self.finbert = pipeline(
                "text-classification", 
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            self.business_classifier = pipeline(
                "text-classification", 
                model="yiyanghkust/finbert-tone",
                return_all_scores=True
            )
            st.success("продолжается пока хорошо: BERT-модели запущены для детекции новостей")
        except Exception as e:
            st.error(f"Ошибка запуска BERT: {str(e)}")
            raise

    def detect_event_type(self, text, entity):
        event_type = "Нет"
        summary = ""
        
        try:
            # Ensure text is properly formatted
            text = str(text).strip()
            if not text:
                return "Нет", "Empty text"

            # Get predictions
            finbert_scores = self.finbert(
                text,
                truncation=True,
                max_length=512
            )
            business_scores = self.business_classifier(
                text,
                truncation=True,
                max_length=512
            )
            
            # Get highest scoring predictions
            finbert_pred = max(finbert_scores[0], key=lambda x: x['score'])
            business_pred = max(business_scores[0], key=lambda x: x['score'])
            
            # Map to event types with confidence threshold
            confidence_threshold = 0.6
            max_confidence = max(finbert_pred['score'], business_pred['score'])
            
            if max_confidence >= confidence_threshold:
                if any(term in text.lower() for term in ['отчет', 'выручка', 'прибыль', 'ebitda']):
                    event_type = "Отчетность"
                    summary = f"Финансовая отчетность (confidence: {max_confidence:.2f})"
                elif any(term in text.lower() for term in ['облигаци', 'купон', 'дефолт', 'реструктуризац']):
                    event_type = "РЦБ"
                    summary = f"Событие РЦБ (confidence: {max_confidence:.2f})"
                elif any(term in text.lower() for term in ['суд', 'иск', 'арбитраж']):
                    event_type = "Суд"
                    summary = f"Судебное разбирательство (confidence: {max_confidence:.2f})"
            
            if event_type != "Нет":
                summary += f"\nКомпания: {entity}"
            
            return event_type, summary
            
        except Exception as e:
            st.warning(f"Event detection error: {str(e)}")
            return "Нет", "Error in event detection"

class TranslationSystem:
    def __init__(self):
        """Initialize translation system using Helsinki NLP model with fallback options"""
        try:
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
            # Initialize fallback translator
            self.fallback_translator = GoogleTranslator(source='ru', target='en')
            self.legacy_translator = LegacyTranslator()
            st.success("начинается все хорошо: запустил систему перевода")
        except Exception as e:
            st.error(f"Ошибка запуска перевода: {str(e)}")
            raise

    def _split_into_chunks(self, text: str, max_length: int = 450) -> list:
        """Split text into chunks while preserving word boundaries"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= max_length:
                current_chunk.append(word)
                current_length += word_length + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _translate_chunk_with_retries(self, chunk: str, max_retries: int = 3) -> str:
        """Attempt translation with multiple fallback options"""
        if not chunk or not chunk.strip():
            return ""

        for attempt in range(max_retries):
            try:
                # First try Helsinki NLP
                result = self.translator(chunk, max_length=512)
                if result and isinstance(result, list) and len(result) > 0:
                    translated = result[0].get('translation_text')
                    if translated and isinstance(translated, str):
                        return translated

                # First fallback: Google Translator
                translated = self.fallback_translator.translate(chunk)
                if translated and isinstance(translated, str):
                    return translated

                # Second fallback: Legacy Google Translator
                translated = self.legacy_translator.translate(chunk, src='ru', dest='en').text
                if translated and isinstance(translated, str):
                    return translated

            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Попробовал перевести {max_retries} раз, не преуспел: {str(e)}")
                time.sleep(1 * (attempt + 1))  # Exponential backoff

        return chunk  # Return original text if all translation attempts fail

    def translate_text(self, text: str) -> str:
        """Translate text with robust error handling and validation"""
        # Input validation
        if pd.isna(text) or not isinstance(text, str):
            return str(text) if pd.notna(text) else ""

        text = str(text).strip()
        if not text:
            return ""

        try:
            # Split into manageable chunks
            chunks = self._split_into_chunks(text)
            translated_chunks = []

            # Process each chunk with validation
            for chunk in chunks:
                if not chunk.strip():
                    continue

                translated_chunk = self._translate_chunk_with_retries(chunk)
                if translated_chunk:  # Only add non-empty translations
                    translated_chunks.append(translated_chunk)
                time.sleep(0.1)  # Rate limiting

            # Final validation of results
            if not translated_chunks:
                return text  # Return original if no translations succeeded

            result = ' '.join(translated_chunks)
            return result if result.strip() else text

        except Exception as e:
            st.warning(f"Translation error: {str(e)}")
            return text  # Return original text on error



def process_file(uploaded_file, model_choice, translation_method=None):
    df = None
    processed_rows_df = pd.DataFrame()

    try:
        # Initialize UI and control systems
        ui = ProcessingUI()
        translator = TranslationSystem()
        event_detector = EventDetectionSystem()
        
        # Load and prepare data
        df = pd.read_excel(uploaded_file, sheet_name='Публикации')
        llm = init_langchain_llm(model_choice)
        
        # Initialize Groq for impact estimation
        groq_llm = ensure_groq_llm()
        if groq_llm is None:
            st.warning("Failed to initialize Groq LLM for impact estimation. Using fallback model.")
        
        # Prepare dataframe
        text_columns = ['Объект', 'Заголовок', 'Выдержки из текста']
        for col in text_columns:
            df[col] = df[col].fillna('').astype(str).apply(lambda x: x.strip())
            
        # Initialize required columns
        df['Translated'] = ''
        df['Sentiment'] = ''
        df['Impact'] = ''
        df['Reasoning'] = ''
        df['Event_Type'] = ''
        df['Event_Summary'] = ''
        
        # Deduplication
        original_count = len(df)
        df = df.groupby('Объект', group_keys=False).apply(
            lambda x: fuzzy_deduplicate(x, 'Выдержки из текста', 65)
        ).reset_index(drop=True)
        st.write(f"Из {original_count} сообщений удалено {original_count - len(df)} дубликатов.")
        
        # Process rows
        total_rows = len(df)
        processed_rows = 0
        
        for idx, row in df.iterrows():
            # Check for stop/pause
            if st.session_state.control.is_stopped():
                st.warning("Обработку остановили")
                if not processed_rows_df.empty:  # Only offer download if we have processed rows
                    output = create_output_file(processed_rows_df, uploaded_file, llm)
                    if output is not None:
                        st.download_button(
                            label=f"📊 Скачать результат ({processed_rows} из {total_rows} строк)",
                            data=output,
                            file_name="partial_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="partial_download"
                        )
                break
                
            st.session_state.control.wait_if_paused()
            if st.session_state.control.is_paused():
                st.info("Обработка на паузе. Можно возобновить.")
                if not processed_rows_df.empty:  # Only offer download if we have processed rows
                    output = create_output_file(processed_rows_df, uploaded_file, llm)
                    if output is not None:
                        st.download_button(
                            label=f"📊 Скачать результат ({processed_rows} из {total_rows} строк)",
                            data=output,
                            file_name="partial_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="partial_download"
                        )
                break                
                continue
                
            try:
                # Translation
                translated_text = translator.translate_text(row['Выдержки из текста'])
                df.at[idx, 'Translated'] = translated_text
                
                # Sentiment analysis
                sentiment = analyze_sentiment(translated_text)
                df.at[idx, 'Sentiment'] = sentiment
                
                # Event detection using BERT/ MT-5
                event_type, event_summary = event_detector.detect_event_type(
                    row['Выдержки из текста'],
                    row['Объект']
                )
                df.at[idx, 'Event_Type'] = event_type
                df.at[idx, 'Event_Summary'] = event_summary
                

                # Show events in real-time
                #if event_type != "Нет":
                #    ui.show_event(
                #        row['Объект'],
                #        event_type,
                #        row['Заголовок']
                #    )
                
                #Calculate processing speed (items per second)
                current_time = time.time()
                time_delta = current_time - last_update_time
                if time_delta > 0:
                    processing_speed = 1 / time_delta  # items per second
                else:
                    processing_speed = 0
                
                # Update live statistics
                ui.update_stats(row, sentiment, event_type, processing_speed)

                
                # Handle negative sentiment
                
                if sentiment == "Negative":
                    try:
                        # Initialize Groq LLM if not already done
                        if 'groq_llm' not in locals():
                        groq_llm = ensure_groq_llm()
            
                        impact, reasoning = estimate_impact(
                        groq_llm if groq_llm is not None else llm,
                        translated_text,
                        row['Объект']
                        )

                    except Exception as e:
                        impact = "Неопределенный эффект"
                        reasoning = "Error in impact estimation"
                        st.warning(f"Impact estimation error: {str(e)}")
                    
                    df.at[idx, 'Impact'] = impact
                    df.at[idx, 'Reasoning'] = reasoning
                    
                    # Show negative alert in real-time
                    #ui.show_negative(
                    #    row['Объект'],
                    #    row['Заголовок'],
                    #    reasoning,
                    #    impact
                    #)
                
                processed_rows_df = pd.concat([processed_rows_df, df.iloc[[idx]]], ignore_index=True)

                # Update progress
                processed_rows += 1
                ui.update_progress(processed_rows, total_rows)
                last_update_time = current_time

            except Exception as e:
                st.warning(f"Ошибка в обработке ряда {idx + 1}: {str(e)}")
                continue
            
            time.sleep(0.1)
        
                
        return processed_rows_df if st.session_state.control.is_stopped() else df
        
    except Exception as e:
        st.error(f"Ошибка в обработке файла: {str(e)}")
        return None


    

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
    try:
        finbert_result = get_mapped_sentiment(
            finbert(text, truncation=True, max_length=512)[0]
        )
        roberta_result = get_mapped_sentiment(
            roberta(text, truncation=True, max_length=512)[0]
        )
        finbert_tone_result = get_mapped_sentiment(
            finbert_tone(text, truncation=True, max_length=512)[0]
        )
        
        # Count occurrences of each sentiment
        sentiments = [finbert_result, roberta_result, finbert_tone_result]
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        
        # Return sentiment if at least two models agree
        for sentiment, count in sentiment_counts.items():
            if count >= 2:
                return sentiment
                
        # Default to Neutral if no agreement
        return "Neutral"
        
    except Exception as e:
        st.warning(f"Sentiment analysis error: {str(e)}")
        return "Neutral"


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
        if model_choice == "Qwen2.5-Coder":
            st.info("Loading Qwen2.5-Coder model. только GPU!")
            return QwenSystem()
            
        elif model_choice == "Groq (llama-3.1-70b)":
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
            
        elif model_choice == "Local-MT5":
            return FallbackLLMSystem()
            
    except Exception as e:
        st.error(f"Error initializing the LLM: {str(e)}")
        st.stop()


def estimate_impact(llm, news_text, entity):
    """
    Estimate impact using Groq LLM regardless of the main model choice.
    Falls back to the provided LLM if Groq initialization fails.
    """
    # Initialize default return values
    impact = "Неопределенный эффект"
    reasoning = "Не удалось получить обоснование"
    
    try:
        # Always try to use Groq first
        groq_llm = ensure_groq_llm()
        working_llm = groq_llm if groq_llm is not None else llm
        
        template = """
        You are a financial analyst. Analyze this news piece about {entity} and assess its potential impact.
        
        News: {news}
        
        Classify the impact into one of these categories:
        1. "Значительный риск убытков" (Significant loss risk)
        2. "Умеренный риск убытков" (Moderate loss risk)
        3. "Незначительный риск убытков" (Minor loss risk)
        4. "Вероятность прибыли" (Potential profit)
        5. "Неопределенный эффект" (Uncertain effect)
        
        Provide a brief, fact-based reasoning for your assessment.
        
        Format your response exactly as:
        Impact: [category]
        Reasoning: [explanation in 2-3 sentences]
        """
        
        prompt = PromptTemplate(template=template, input_variables=["entity", "news"])
        chain = prompt | working_llm
        
        # Handle both dictionary and direct string inputs
        if isinstance(news_text, dict):
            response = chain.invoke(news_text)
        else:
            response = chain.invoke({"entity": entity, "news": news_text})
        
        # Extract content from response
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        if "Impact:" in response_text and "Reasoning:" in response_text:
            try:
                impact_part, reasoning_part = response_text.split("Reasoning:")
                impact_temp = impact_part.split("Impact:")[1].strip()
                
                # Validate impact category
                valid_impacts = [
                    "Значительный риск убытков",
                    "Умеренный риск убытков",
                    "Незначительный риск убытков",
                    "Вероятность прибыли",
                    "Неопределенный эффект"
                ]
                
                # Use fuzzy matching to handle slight variations
                best_match = None
                best_score = 0
                impact_temp_lower = impact_temp.lower()
                
                for valid_impact in valid_impacts:
                    score = fuzz.ratio(impact_temp_lower, valid_impact.lower())
                    if score > best_score and score > 80:  # 80% similarity threshold
                        best_score = score
                        best_match = valid_impact
                
                impact = best_match if best_match else "Неопределенный эффект"
                reasoning = reasoning_part.strip()
                
            except Exception as e:
                st.warning(f"Error parsing impact response: {str(e)}")
                
    except Exception as e:
        st.warning(f"Error in impact estimation: {str(e)}")
        if 'rate limit' in str(e).lower():
            st.warning("Rate limit reached. Using fallback model.")
    
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

def translate_reasoning_to_russian(llm, text):
    """Modified to handle both standard LLMs and FallbackLLMSystem"""
    if isinstance(llm, FallbackLLMSystem):
        # Direct translation using MT5
        response = llm.invoke({
            'template_result': f"Translate to Russian: {text}"
        })
        return response.content.strip()
    else:
        # Original LangChain approach
        template = """
        Translate this English explanation to Russian, maintaining a formal business style:
        "{text}"
        
        Your response should contain only the Russian translation.
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        chain = prompt | llm
        response = chain.invoke({"text": text})
        
        # Handle different response types
        if hasattr(response, 'content'):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response).strip()

def create_output_file(df, uploaded_file, llm):
    try:
        wb = load_workbook("sample_file.xlsx")
        
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

               
        # Update 'Анализ' sheet with modified translation handling
        ws = wb['Анализ']
        row_idx = 4
        for _, row in df[df['Sentiment'] == 'Negative'].iterrows():
            ws.cell(row=row_idx, column=5, value=row['Объект'])
            ws.cell(row=row_idx, column=6, value=row['Заголовок'])
            ws.cell(row=row_idx, column=7, value="Риск убытка")
            
            # Enhanced translation handling
            if pd.notna(row['Reasoning']):
                try:
                    grlm = init_langchain_llm("Groq (llama-3.1-70b)")
                    translated_reasoning = translate_reasoning_to_russian(grlm, row['Reasoning'])
                    ws.cell(row=row_idx, column=8, value=translated_reasoning)
                except Exception as e:
                    st.warning(f"Translation error for row {row_idx}: {str(e)}")
                    ws.cell(row=row_idx, column=8, value=row['Reasoning'])  # Use original text as fallback
            
            ws.cell(row=row_idx, column=9, value=row['Выдержки из текста'])
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
        
    except Exception as e:
        st.warning(f"Ошибка при создании выходного файла: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    
    with st.sidebar:
        st.title("::: AI-анализ мониторинга новостей (v.3.70):::")
        st.subheader("по материалам СКАН-ИНТЕРФАКС")
        
        model_choice = st.radio(
            "Выберите модель для анализа:",
            ["Local-MT5", "Qwen2.5-Coder", "Groq (llama-3.1-70b)", "ChatGPT-4-mini"],
            key="model_selector",
            help="Выберите модель для анализа новостей"
        )
        
        uploaded_file = st.file_uploader(
            "Выбирайте Excel-файл",
            type="xlsx",
            key="file_uploader"
        )
        
        st.markdown(
            """
            Использованы технологии:  
            - Анализ естественного языка с помощью предтренированных нейросетей **BERT**
            - Дополнительная обработка при помощи больших языковых моделей (**LLM**)
            - Фреймворк **LangChain** для оркестрации
            """,
            unsafe_allow_html=True
        )

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

    # Main content area
    st.title("Анализ мониторинга новостей")
    
    # Initialize session state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
        
    # Create display areas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Area for real-time updates
        st.subheader("Что найдено, показываю:")
        st.markdown("""
            <style>
            .stProgress .st-bo {
                background-color: #f0f2f6;
            }
            .negative-alert {
                background-color: #ffebee;
                border-left: 5px solid #f44336;
                padding: 10px;
                margin: 5px 0;
            }
            .event-alert {
                background-color: #e3f2fd;
                border-left: 5px solid #2196f3;
                padding: 10px;
                margin: 5px 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
    with col2:
        # Area for statistics
        st.subheader("Статистика")
        if st.session_state.processed_df is not None:
            st.metric("Всего статей", len(st.session_state.processed_df))
            st.metric("Из них негативных", 
                len(st.session_state.processed_df[
                    st.session_state.processed_df['Sentiment'] == 'Negative'
                ])
            )
            st.metric("Событий обнаружено", 
                len(st.session_state.processed_df[
                    st.session_state.processed_df['Event_Type'] != 'Нет'
                ])
            )
    
    if uploaded_file is not None and st.session_state.processed_df is None:
        start_time = time.time()
        
        try:
            st.session_state.processed_df = process_file(
                uploaded_file,
                model_choice,
                translation_method='auto'
            )
            
            if st.session_state.processed_df is not None:
                end_time = time.time()
                elapsed_time = format_elapsed_time(end_time - start_time)
                
                # Show results
                st.subheader("Итого по результатам")
                
                # Display statistics
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("Всего обработано", len(st.session_state.processed_df))
                with stats_cols[1]:
                    st.metric("Негативных", 
                        len(st.session_state.processed_df[
                            st.session_state.processed_df['Sentiment'] == 'Negative'
                        ])
                    )
                with stats_cols[2]:
                    st.metric("Событий обнаружено", 
                        len(st.session_state.processed_df[
                            st.session_state.processed_df['Event_Type'] != 'Нет'
                        ])
                    )
                with stats_cols[3]:
                    st.metric("Время обработки составило", elapsed_time)
                
                # Show data previews
                with st.expander("📊 Предпросмотр данных", expanded=True):
                    preview_cols = ['Объект', 'Заголовок', 'Sentiment', 'Event_Type']
                    st.dataframe(
                        st.session_state.processed_df[preview_cols],
                        use_container_width=True
                    )
                
                # Create downloadable report
                output = create_output_file(
                    st.session_state.processed_df,
                    uploaded_file,
                    init_langchain_llm(model_choice)
                )
                
                st.download_button(
                    label="📥 Полный отчет - загрузить",
                    data=output,
                    file_name="результаты_анализа.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='download_button'
                )
                
        except Exception as e:
            st.error(f"Ошибочка в обработке файла: {str(e)}")
            st.session_state.processed_df = None


if __name__ == "__main__":
    main()