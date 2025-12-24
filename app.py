import streamlit as st
import time
import pandas as pd
import numpy as np
import PyPDF2
import io
import requests
import json
import base64

# Try to import plotly, fallback to bar_chart if not available
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# PDF Extraction Helper
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Agent 1: Context Architect helper (Compatible with Python 3.8+)
def generate_interview_questions(resume_text, job_description, api_key):
    try:
        # We prioritize the hardcoded key, then the one from sidebar
        hardcoded_key = "AIzaSyB_qCCWltHfCTldS0MLAe7mx_JFF_AXII8".strip()
        FINAL_API_KEY = hardcoded_key if hardcoded_key else api_key.strip()
        
        system_instruction = """
        You are an elite Technical Recruiter at a Fortune 500 company. 
        Analyze the provided Resume and Job Description. 
        Identify the most critical technical gaps or strengths. 
        Generate 6 unique, challenging interview questions:
        - 3 technical deep-dives into their past projects.
        - 2 behavioral/situational questions.
        - 1 "Charisma/Leadership" question.
        Output MUST be a numbered list (1. 2. 3. 4. 5. 6.).
        """
        
        prompt = f"{system_instruction}\n\nRESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
        
        # Models to try (User's futuristic choices + stable fallbacks)
        models_to_try = [
            "gemini-3-flash-preview", 
            "gemini-1.5-flash", 
            "gemini-1.5-pro"
        ]
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
        }

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={FINAL_API_KEY}"
            
            for attempt in range(2): # Simple retry for 429
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        text_response = result['candidates'][0]['content']['parts'][0]['text']
                        
                        questions = []
                        for line in text_response.split('\n'):
                            line = line.strip()
                            if line and ((line[0].isdigit() and (len(line) > 1 and (line[1] == '.' or line[1] == ')'))) or (len(line) > 2 and line[0].isdigit() and line[2] == '.')):
                                content = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                                if content: questions.append(content)
                        
                        if len(questions) >= 6: return questions[:6]
                        elif len(questions) > 0: return questions
                        continue # Try next model if parsing failed
                        
                    elif response.status_code == 429:
                        st.toast(f"Agent 1 is busy... Retrying in 5s", icon="⏳")
                        time.sleep(5)
                        continue
                    elif response.status_code == 404:
                        break # Model not found, try next model in outer loop
                    else:
                        break # Other error, try next model
                except Exception:
                    break # Connection error, try next model
                    
        return None
    except Exception as e:
        st.error(f"AI Generation Error: {e}")
        return None

# Agent 2: Multimodal Coach helper (Revised Phase 3)
def analyze_interview_delivery(media_file, question_text):
    FINAL_API_KEY = "AIzaSyB_qCCWltHfCTldS0MLAe7mx_JFF_AXII8"
    
    # Read and encode media
    media_bytes = media_file.read()
    media_base64 = base64.b64encode(media_bytes).decode('utf-8')
    mime_type = media_file.type if hasattr(media_file, 'type') else "image/jpeg"
    
    system_instruction = f"""
    You are a high-performance Charisma Coach. Analyze this interview response.
    Focus on: 
    1) Eye contact consistency. 
    2) Facial engagement (smiling/warmth vs. stress). 
    3) Body posture (open vs. closed).
    
    Compare their behavior to the difficulty of the Question: {question_text}.
    
    Provide a Confidence Score out of 100, Eye Contact Score (0-100), Posture Score (0-100), and Tone Score (0-100).
    Also provide three specific, timestamped improvement points.
    
    Return the result as a JSON object with keys: 
    'confidence', 'eye_contact', 'posture', 'tone', 'feedback'.
    """
    
    # Models to try (User's preferred models first)
    models_to_try = [
        "gemini-3-flash", 
        "gemini-3-flash-preview", 
        "gemini-1.5-flash", 
        "gemini-2.5-flash-latest",
        "gemini-pro"
    ]
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [
                {"text": system_instruction},
                {"inline_data": {
                    "mime_type": mime_type, 
                    "data": media_base64
                }}
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "response_mime_type": "application/json",
            "candidate_count": 1
        }
    }

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={FINAL_API_KEY}"
        
        for attempt in range(3): # Implement 5s retry loop as requested
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=45)
                
                if response.status_code == 200:
                    result = response.json()
                    raw_text = result['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(raw_text)
                
                elif response.status_code == 429:
                    st.toast("Coach is reviewing your footage...", icon="⏳")
                    time.sleep(5)
                    continue
                
                elif response.status_code == 404:
                    break # Model not found, try next model
                
                else:
                    break # Other error, try next model
                    
            except Exception:
                break # Connection error, try next model
                
    st.error("Failed to analyze delivery with any compatible model.")
    return None

# Page configuration
st.set_page_config(
    page_title="CharismaAI - Interview Mastery",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Enterprise Dark Theme with Orange Accents
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    /* Center header */
    .header-container {
        text-align: center;
        padding: 2rem 0;
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .tagline {
        font-size: 1.2rem;
        color: #f7931e;
        font-weight: 400;
        margin-top: -0.5rem;
        letter-spacing: 1px;
    }
    
    /* Box Styling */
    .setup-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(247, 147, 30, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .setup-box:hover {
        border: 1px solid rgba(247, 147, 30, 0.5);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        margin-top: 2rem;
    }
    
    /* Highlighted Question */
    .question-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        line-height: 1.4;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 2.5rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.2);
        width: auto !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
        background: linear-gradient(90deg, #f7931e 0%, #ff6b35 100%);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #f7931e !important;
    }
    
    /* Custom divider */
    .orange-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff6b35, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'stage' not in st.session_state:
    st.session_state.stage = 'SETUP'
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'recordings' not in st.session_state:
    st.session_state.recordings = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# --- SIDEBAR (API Configuration) ---
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")
    st.markdown("---")
    st.info("Your API key is only used for the current session and is not stored.")

# --- HEADER (Visible on all stages) ---
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">CharismaAI</h1>
        <p class="tagline">Your AI-Powered Path to Interview Mastery</p>
    </div>
""", unsafe_allow_html=True)

# --- STAGE 1: SETUP ---
if st.session_state.stage == 'SETUP':
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="setup-box">', unsafe_allow_html=True)
        st.subheader("📄 Resume Analysis")
        uploaded_resume = st.file_uploader("Upload Resume (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
        resume_text = st.text_area("Or Paste Resume Content", height=200, placeholder="Paste your resume text here...")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="setup-box">', unsafe_allow_html=True)
        st.subheader("💼 Job Description")
        job_desc = st.text_area("Target Job Description", height=288, placeholder="Paste the job requirements and description here...")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Generate 5-6 Interview Questions", use_container_width=True):
            if (uploaded_resume or resume_text) and job_desc:
                with st.spinner("Agent 1 is analyzing your profile and generating your interview roadmap..."):
                    # 1. Extract Resume Text
                    final_resume_text = resume_text
                    if uploaded_resume:
                        pdf_text = extract_text_from_pdf(uploaded_resume)
                        if pdf_text:
                            final_resume_text = f"{resume_text}\n\n[EXTRACTED FROM PDF]:\n{pdf_text}"
                    
                    # 2. Generate Questions
                    generated_questions = generate_interview_questions(final_resume_text, job_desc, gemini_api_key)
                    
                    if generated_questions:
                        st.session_state.questions = generated_questions
                        st.session_state.stage = 'INTERVIEW'
                        st.session_state.current_idx = 0
                        st.success("Questions generated successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to generate questions. Please check your AI configuration or inputs.")
            else:
                st.warning("Please provide both a Resume (PDF or text) and a Job Description.")

# --- STAGE 2: INTERVIEW ---
elif st.session_state.stage == 'INTERVIEW':
    q_idx = st.session_state.current_idx
    total_q = len(st.session_state.questions)
    
    st.markdown(f"#### Question {q_idx + 1} of {total_q}")
    st.markdown(f'<div class="question-text">{st.session_state.questions[q_idx]}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("##### 📤 Option 1: Full Video Analysis")
        st.caption("Upload a recorded video (mp4, mov, avi) for deep behavioral analysis.")
        uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"], key=f"upload_{q_idx}", help="Best for full analysis of eye contact, pacing, and tone.")
        if uploaded_video:
            st.session_state.recordings[q_idx] = "uploaded"
            st.video(uploaded_video)
            st.success("✅ Video ready for analysis")
            
    with rec_col2:
        st.markdown("##### 🔴 Option 2: Quick Delivery Snapshot")
        st.caption("Take a snapshot to analyze your posture, smile, and immediate eye contact.")
        live_capture = st.camera_input("Capture Response Preview", key=f"live_{q_idx}", help="Note: Direct video recording is not supported. Use Option 1 for full videos.")
        if live_capture:
            st.session_state.recordings[q_idx] = "live"
            st.image(live_capture, caption="Snapshot captured", use_container_width=True)
            st.success("✅ Snapshot ready for analysis")
            
    # --- ANALYSIS TRIGGER ---
    st.markdown("---")
    analysis_ready = q_idx in st.session_state.recordings
    analysis_done = q_idx in st.session_state.analysis_results
    
    if analysis_ready and not analysis_done:
        if st.button("🚀 Analyze My Response", use_container_width=True):
            with st.spinner("Agent 2 is analyzing your delivery..."):
                # Get the media file
                media_file = uploaded_video if st.session_state.recordings[q_idx] == "uploaded" else live_capture
                
                result = analyze_interview_delivery(media_file, st.session_state.questions[q_idx])
                if result:
                    st.session_state.analysis_results[q_idx] = result
                    st.success("Analysis complete! Proceed to results or next question.")
                    st.rerun()
    elif analysis_done:
        st.success("✅ Delivery analyzed by Agent 2")
        with st.expander("Peek at Coach's Notes"):
            res = st.session_state.analysis_results[q_idx]
            st.write(f"**Confidence Score:** {res.get('confidence', 'N/A')}")
            for point in res.get('feedback', []):
                st.write(f"- {point}")

    st.markdown("---")
    st.markdown("<br><br>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
    
    with nav_col1:
        if st.button("← Previous", disabled=(q_idx == 0)):
            st.session_state.current_idx -= 1
            st.rerun()
            
    with nav_col3:
        if q_idx == total_q - 1:
            if st.button("Finish →"):
                st.session_state.stage = 'RESULTS'
                st.rerun()
        else:
            if st.button("Next →"):
                st.session_state.current_idx += 1
                st.rerun()

# --- STAGE 3: RESULTS ---
elif st.session_state.stage == 'RESULTS':
    st.markdown('<div class="main-title" style="font-size: 2.5rem; text-align: center;">Final Analysis Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
    
    # --- RESULTS CALCULATIONS ---
    results_count = len(st.session_state.analysis_results)
    if results_count > 0:
        all_res = list(st.session_state.analysis_results.values())
        avg_confidence = np.mean([float(r.get('confidence', 0)) for r in all_res])
        avg_eye = np.mean([float(r.get('eye_contact', 0)) for r in all_res])
        avg_posture = np.mean([float(r.get('posture', 0)) for r in all_res])
        avg_tone = np.mean([float(r.get('tone', 0)) for r in all_res])
        
        metrics = {
            "Confidence": int(avg_confidence), 
            "Eye Contact": int(avg_eye), 
            "Posture": int(avg_posture), 
            "Tone & Delivery": int(avg_tone)
        }
    else:
        metrics = {"Confidence": 0, "Eye Contact": 0, "Posture": 0, "Tone & Delivery": 0}

    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Confidence", f"{metrics['Confidence']}%")
    with col2: st.metric("Eye Contact", f"{metrics['Eye Contact']}%")
    with col3: st.metric("Posture", f"{metrics['Posture']}%")
    with col4: st.metric("Tone & Delivery", f"{metrics['Tone & Delivery']}%")

    st.markdown("---")
    
    # Visualization
    st.markdown("### 📈 Performance Overview")
    if HAS_PLOTLY:
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()), 
                y=list(metrics.values()),
                marker_color=['#FF8C00', '#FFA500', '#FFB347', '#FFCC33']
            )
        ])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_data = pd.DataFrame({'Metric': list(metrics.keys()), 'Score': list(metrics.values())})
        st.bar_chart(chart_data, x='Metric', y='Score', color="#FF8C00")

    # Detailed Breakdown Table
    st.markdown("### 📝 Detailed Performance Breakdown")
    data = []
    for i, q in enumerate(st.session_state.questions):
        analysis = st.session_state.analysis_results.get(i, {})
        feedback_points = analysis.get('feedback', ["No feedback available."])
        # Join feedback points with bullets
        formatted_feedback = " ".join([f"• {p}" for p in feedback_points]) if isinstance(feedback_points, list) else feedback_points
        
        data.append({
            "Question": f"Q{i+1}: {q[:50]}...",
            "Confidence": f"{analysis.get('confidence', 0)}%",
            "Coach Feedback": formatted_feedback
        })
    
    df = pd.DataFrame(data)
    st.table(df)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start New Session"):
        st.session_state.stage = 'SETUP'
        st.session_state.recordings = {}
        st.session_state.questions = []
        st.rerun()

# --- FOOTER ---
st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 0.85rem; padding: 1rem;'>
        CharismaAI Enterprise | Visualizing Professional Growth
    </div>
""", unsafe_allow_html=True)
