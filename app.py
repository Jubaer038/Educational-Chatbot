from __future__ import annotations

import datetime as dt

from dotenv import load_dotenv
import streamlit as st

from config import prompts, settings
from services import essay_writer, evaluator, pdf_chatbot, planner, summarizer
from utils import validators

load_dotenv()
st.set_page_config(page_title="Educational Chatbot - AI Learning Assistant", layout="wide")
st.title("🎓 Educational Chatbot - AI Learning Assistant")
st.caption(
    "Upload PDFs, ask questions with source citations, get rubric-based evaluation, and access study planning tools."
)

with st.sidebar:
    st.markdown("**🤖 AI Model**")
    st.write(f"Provider: **Groq Cloud**")
    st.write(f"Model: `{settings.GROQ_MODEL}`")
    st.success("✓ Free-tier cloud inference")
    
    st.markdown("---")
    st.markdown("**📄 PDF Processing**")
    st.write(f"Chunk size: {settings.CHUNK_SIZE}")
    st.write(f"Overlap: {settings.CHUNK_OVERLAP}")
    st.write(f"Top-K retrieval: {settings.TOP_K}")


tab_qa, tab_plan, tab_essay, tab_sum = st.tabs(
    ["PDF Q&A", "Study Planner", "Essay Writer", "Summarizer"]
)


with tab_qa:
    st.subheader("📖 Educational Chatbot - PDF Q&A with Source Citations")
    st.info("💡 Upload any PDF and ask questions. Get answers grounded in the document with exact page numbers, sections, and text snippets.")
    
    pdf_file = st.file_uploader("Upload PDF Document", type=["pdf"], key="qa_pdf")
    question = st.text_input("Your Question:", placeholder="e.g., What are the main concepts in Chapter 3?", key="qa_question")
    
    # Initialize session state to store Q&A for evaluation (use different keys to avoid conflict)
    if 'stored_qa_result' not in st.session_state:
        st.session_state.stored_qa_result = None
    if 'stored_qa_question' not in st.session_state:
        st.session_state.stored_qa_question = None
    
    if st.button("🔍 Get Answer", type="primary"):
        file_bytes = validators.validate_pdf(pdf_file)
        if file_bytes and question:
            with st.spinner("Analyzing PDF and generating answer..."):
                result = pdf_chatbot.answer_question(question, file_bytes, pdf_file.name)
            
            # Store in session state for evaluation
            st.session_state.stored_qa_result = result
            st.session_state.stored_qa_question = question
            
            st.markdown("### ✅ Answer")
            st.success(result.get("answer", ""))
            
            if result.get("citations"):
                st.markdown("### 📄 Source Citations")
                st.caption("These excerpts from the PDF support the answer above:")
                for idx, cit in enumerate(result["citations"], 1):
                    with st.expander(f"📌 Citation {idx}: Page {cit['page']} (Relevance: {cit['score']:.2f})"):
                        st.write(cit['snippet'])
        elif not question:
            st.warning("⚠️ Please enter a question to ask about the PDF.")
    
    # Display stored answer if exists
    elif st.session_state.stored_qa_result:
        st.markdown("### ✅ Answer")
        st.success(st.session_state.stored_qa_result.get("answer", ""))
        
        if st.session_state.stored_qa_result.get("citations"):
            st.markdown("### 📄 Source Citations")
            st.caption("These excerpts from the PDF support the answer above:")
            for idx, cit in enumerate(st.session_state.stored_qa_result["citations"], 1):
                with st.expander(f"📌 Citation {idx}: Page {cit['page']} (Relevance: {cit['score']:.2f})"):
                    st.write(cit['snippet'])
    
    st.markdown("---")
    st.markdown("### ✅ Answer Evaluation - Automated Rubric-Based Scoring")
    st.info("💯 Got an answer above? Now evaluate it against the PDF reference with detailed feedback.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 📊 Evaluation Rubric")
        st.markdown("""
        - **Concept Correctness** (40%): Alignment with reference answer
        - **Coverage** (30%): Completeness of key points
        - **Clarity** (20%): Sentence structure and readability
        - **Language** (10%): Grammar and punctuation
        """)
    
    with col2:
        st.markdown("#### 📝 Scoring Guide")
        st.markdown("""
        - **90-100**: Excellent understanding
        - **70-89**: Good understanding
        - **50-69**: Adequate, needs improvement
        - **Below 50**: Requires significant revision
        """)
    
    # Show current Q&A for evaluation
    if st.session_state.stored_qa_question and st.session_state.stored_qa_result:
        st.text_input("Question being evaluated:", value=st.session_state.stored_qa_question, disabled=True, key="eval_q_display")
        st.text_area("AI Answer to evaluate:", value=st.session_state.stored_qa_result.get("answer", ""), disabled=True, height=100, key="eval_a_display")
        st.caption("⬆️ This is the AI-generated answer that will be evaluated against the PDF reference")
    else:
        st.warning("⚠️ Please get an answer first using the 'Get Answer' button above, then you can evaluate it.")
    
    if st.button("📊 Evaluate AI Answer", type="primary", key="eval_btn", disabled=not (st.session_state.stored_qa_question and st.session_state.stored_qa_result)):
        file_bytes = validators.validate_pdf(pdf_file)
        if file_bytes and st.session_state.stored_qa_question and st.session_state.stored_qa_result:
            with st.spinner("Building reference answer from PDF and evaluating the AI response..."):
                ref = pdf_chatbot.reference_answer(st.session_state.stored_qa_question, file_bytes, pdf_file.name)
                ai_answer = st.session_state.stored_qa_result.get("answer", "")
                grading = evaluator.grade_response(st.session_state.stored_qa_question, ai_answer, ref)
            
            # Score display
            score = grading['score']
            if score >= 90:
                st.success(f"### 🌟 Excellent! Score: {score} / 100")
            elif score >= 70:
                st.info(f"### 👍 Good! Score: {score} / 100")
            elif score >= 50:
                st.warning(f"### 📈 Adequate. Score: {score} / 100")
            else:
                st.error(f"### 📉 Needs Improvement. Score: {score} / 100")
            
            # Detailed breakdown
            st.markdown("#### 📋 Detailed Rubric Breakdown")
            details = grading.get("details", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Concept", f"{details.get('concept_correctness', 0):.1f}%", help="40% weight")
            with col2:
                st.metric("Coverage", f"{details.get('coverage', 0):.1f}%", help="30% weight")
            with col3:
                st.metric("Clarity", f"{details.get('clarity', 0):.1f}%", help="20% weight")
            with col4:
                st.metric("Language", f"{details.get('language', 0):.1f}%", help="10% weight")
            
            # Feedback
            if grading.get("strengths"):
                st.markdown("#### ✨ Strengths")
                for strength in grading["strengths"]:
                    st.success(f"✓ {strength}")
            
            if grading.get("improvements"):
                st.markdown("#### 💡 Areas for Improvement")
                for improvement in grading["improvements"]:
                    st.warning(f"→ {improvement}")
            
            # Reference answer
            if ref:
                with st.expander("📖 View Reference Answer (from PDF)"):
                    st.info(ref)
        elif not pdf_file:
            st.warning("⚠️ Please upload a PDF document first.")


with tab_plan:
    st.subheader("Exam Study Planner")
    exam_date = st.date_input("Exam date", min_value=dt.date.today() + dt.timedelta(days=1))
    daily_hours = st.number_input("Available hours per day", min_value=1.0, max_value=12.0, value=3.0, step=0.5)
    st.markdown("Enter topics as `Topic | difficulty(1-5) | priority(1-5)` one per line")
    topics_raw = st.text_area(
        "Topics",
        value="Algebra | 4 | 5\nCalculus | 5 | 5\nHistory | 2 | 2",
        height=120,
    )
    if st.button("Create Plan", type="primary"):
        if validators.validate_date(exam_date):
            topics = planner.parse_topics(topics_raw)
            if not topics:
                st.error("Add at least one topic with difficulty and priority.")
            else:
                plan = planner.build_plan(exam_date, daily_hours, topics)
                st.write(f"Plan for {len(plan)} days")
                for day in plan:
                    formatted = day["day"].strftime("%Y-%m-%d (%a)")
                    st.markdown(f"**{formatted}**")
                    for task in day["tasks"]:
                        st.write(f"- {task}")


with tab_essay:
    st.subheader("AI Essay Writer")
    topic = st.text_input("Topic")
    word_limit = st.number_input("Word limit", min_value=150, max_value=2000, value=400, step=50)
    tone = st.selectbox("Tone", ["Academic", "Friendly", "Persuasive", "Narrative"], index=0)
    outline = st.text_area("Optional outline")
    if st.button("Generate Essay", type="primary"):
        if topic:
            with st.spinner("Writing essay..."):
                essay = essay_writer.write_essay(topic, int(word_limit), tone, outline)
            st.write(essay)
        else:
            st.warning("Please enter a topic.")


with tab_sum:
    st.subheader("AI Text & PDF Summarization")
    mode = st.radio("Mode", options=["short", "bullets"], horizontal=True)
    source = st.radio("Source", options=["Text", "PDF"], horizontal=True)
    if source == "Text":
        text_content = st.text_area("Paste text to summarize", height=200)
        if st.button("Summarize Text", type="primary"):
            if text_content:
                with st.spinner("Summarizing..."):
                    summary = summarizer.summarize_text(text_content, mode=mode)  # type: ignore[arg-type]
                st.write(summary)
            else:
                st.warning("Please paste some text.")
    else:
        sum_pdf = st.file_uploader("Upload PDF to summarize", type=["pdf"], key="sum_pdf")
        if st.button("Summarize PDF", type="primary"):
            file_bytes = validators.validate_pdf(sum_pdf)
            if file_bytes:
                with st.spinner("Summarizing PDF..."):
                    summary = summarizer.summarize_pdf(file_bytes, mode=mode)  # type: ignore[arg-type]
                st.write(summary)

st.markdown("---")
st.caption("All prompts are versioned in `config/prompts.py`. Cached embeddings per PDF are managed with Streamlit cache.")
