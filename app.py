# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import joblib
import re

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Emotion Intelligence Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
    <style>
    /* Main header gradient */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    
    /* Emotion cards */
    .emotion-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .emotion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background: #f7fafc;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Emotion color classes */
    .anger-color { color: #EF4444; }
    .disgust-color { color: #10B981; }
    .fear-color { color: #F59E0B; }
    .happy-color { color: #EC4899; }
    .joy-color { color: #F59E0B; }
    .neutral-color { color: #6B7280; }
    .sad-color { color: #3B82F6; }
    .sadness-color { color: #3B82F6; }
    .shame-color { color: #8B5CF6; }
    .surprise-color { color: #8B5CF6; }
    </style>
""", unsafe_allow_html=True)

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    """Load emotion and depression models"""
    try:
        # Load emotion models
        emotion_model = joblib.load(r"models\emotion_model.pkl")
        emotion_vectorizer = joblib.load(r"models\emotion_vectorizer.pkl")
        emotion_le = joblib.load(r"models\emotion_label_encoder.pkl")
        
        # Load depression models
        depression_model = joblib.load(r"models\depression_model.pkl")
        depression_vectorizer = joblib.load(r"models\depression_vectorizer.pkl")
        
        st.sidebar.success("✅ All models loaded successfully!")
        return (emotion_model, emotion_vectorizer, emotion_le, 
                depression_model, depression_vectorizer)
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        return None

# ====================== EMOTION CONFIGURATION ======================
emotion_colors = {
    'anger': '#EF4444',      # Red
    'disgust': '#10B981',    # Green
    'fear': '#F59E0B',       # Amber/Orange
    'happy': '#EC4899',      # Pink
    'joy': '#F59E0B',        # Orange
    'neutral': '#6B7280',    # Gray
    'sad': '#3B7280',        # Blue-Gray
    'sadness': '#3B82F6',    # Blue
    'shame': '#8B5CF6',      # Purple
    'surprise': '#8B5CF6',   # Purple
}

emotion_icons = {
    'anger': '😠',
    'disgust': '🤮', 
    'fear': '😨',
    'happy': '😊',
    'joy': '😂',
    'neutral': '😐',
    'sad': '😔',
    'sadness': '😔',
    'shame': '😳',
    'surprise': '😮',
}

emotion_descriptions = {
    'anger': 'Anger, frustration, irritation, rage, and annoyance',
    'disgust': 'Disgust, revulsion, repulsion, and contempt',
    'fear': 'Fear, anxiety, worry, nervousness, and terror',
    'happy': 'Happiness, joy, contentment, pleasure, and delight',
    'joy': 'Joy, amusement, laughter, and exhilaration',
    'neutral': 'Neutral, indifferent, unemotional, and objective',
    'sad': 'Sadness, sorrow, grief, disappointment, and depression',
    'sadness': 'Sadness, sorrow, grief, disappointment, and depression',
    'shame': 'Shame, embarrassment, guilt, and humiliation',
    'surprise': 'Surprise, astonishment, amazement, and shock',
}

# ====================== ANALYSIS FUNCTIONS ======================
def predict_emotion(text, emotion_model, emotion_vectorizer, emotion_le):
    """Predict emotion for text"""
    text_vec = emotion_vectorizer.transform([text])
    emotion_probs = emotion_model.predict_proba(text_vec)[0]
    emotion_index = np.argmax(emotion_probs)
    emotion_label = emotion_le.inverse_transform([emotion_index])[0]
    return emotion_label, emotion_probs, emotion_probs[emotion_index]

def predict_depression(text, depression_model, depression_vectorizer):
    """Predict depression for text"""
    text_vec = depression_vectorizer.transform([text])
    dep_probs = depression_model.predict_proba(text_vec)[0]
    dep_pred = np.argmax(dep_probs)
    return dep_pred, dep_probs, dep_probs[dep_pred]

def analyze_text(text, models):
    """Comprehensive text analysis"""
    if not text.strip() or models is None:
        return None
    
    # Unpack models
    emotion_model, emotion_vectorizer, emotion_le, depression_model, depression_vectorizer = models
    
    # Get predictions
    emotion_label, emotion_probs, emotion_confidence = predict_emotion(text, emotion_model, emotion_vectorizer, emotion_le)
    dep_pred, dep_probs, dep_confidence = predict_depression(text, depression_model, depression_vectorizer)
    
    # Create emotion probabilities list
    emotion_probs_list = []
    for idx, prob in enumerate(emotion_probs):
        emotion = emotion_le.classes_[idx]
        emotion_probs_list.append({
            'emotion': emotion,
            'probability': float(prob),
            'icon': emotion_icons.get(emotion, ''),
            'color': emotion_colors.get(emotion, '#6B7280')
        })
    
    # Sort by probability
    emotion_probs_list.sort(key=lambda x: x['probability'], reverse=True)
    
    # Calculate emotional intensity
    emotional_intensity = float(np.max(emotion_probs))
    significant_emotions = sum(1 for ep in emotion_probs_list if ep['probability'] > 0.1)
    emotional_complexity = "Simple" if significant_emotions <= 2 else "Complex"
    
    # Depression status
    depression_status = "Detected" if dep_pred == 1 else "Not Detected"
    depression_color = "#EF4444" if dep_pred == 1 else "#10B981"
    depression_icon = "⚠️" if dep_pred == 1 else "✅"
    
    return {
        'text': text,
        'primary_emotion': emotion_label,
        'primary_probability': float(emotion_confidence),
        'emotion_probabilities': emotion_probs_list,
        'emotional_intensity': emotional_intensity,
        'emotional_complexity': emotional_complexity,
        'depression_status': depression_status,
        'depression_confidence': float(dep_confidence),
        'depression_color': depression_color,
        'depression_icon': depression_icon,
        'timestamp': datetime.now(),
        'top_emotions': [ep['emotion'] for ep in emotion_probs_list[:3]]
    }

# ====================== MAIN APPLICATION ======================
def main():
    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=100)
        
        st.markdown("### 🎯 About This Model")
        
        # Load models
        models = load_models()
        
        if models is not None:
            emotion_model, emotion_vectorizer, emotion_le, _, _ = models
            emotions_list = list(emotion_le.classes_)
            st.info(f"""
                **Dual Model System:**
                - **Emotion Detection:** {len(emotions_list)} categories
                - **Depression Detection:** Binary classification
                - **Real-time analysis**
                
                **Emotions Detected:**
                {', '.join([emotion_icons.get(e, '') + ' ' + e.title() for e in emotions_list[:4]])}...
            """)
        
        st.markdown("---")
        st.markdown("### 🎭 Emotion Guide")
        
        # Show first 4 emotions in guide
        if models is not None:
            emotion_model, emotion_vectorizer, emotion_le, _, _ = models
            for emotion in list(emotion_le.classes_)[:4]:
                with st.expander(f"{emotion_icons.get(emotion, '')} {emotion.title()}"):
                    st.caption(emotion_descriptions.get(emotion, ''))
        
        st.markdown("---")
        st.markdown("### 🚀 How to Use")
        st.info("""
        1. Go to **Analyze** tab
        2. Enter or paste your text
        3. Click **Analyze Emotions**
        4. View detailed results for both emotion and depression
        5. Check history in **Dashboard**
        """)
    
    # ====================== MAIN CONTENT ======================
    # Title and description
    st.markdown('<h1 class="main-header">🧠 Emotion & Depression Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #4a5568; margin-bottom: 3rem; font-size: 1.1rem; line-height: 1.6;'>
            Advanced dual-analysis system for detecting emotions and depression risk in text. 
            Real-time analysis with comprehensive insights.
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📈 Dashboard", "📚 About"])
    
    # ====================== TAB 1: ANALYZE ======================
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📝 Enter Text for Analysis")
            
            # Sample text buttons
            st.markdown("#### Try Sample Texts:")
            
            sample_texts = {
                "Angry": "I'm furious about how they treated me! This is completely unfair and unacceptable!",
                "Happy": "I'm absolutely thrilled! Just got promoted at work and everything is going perfectly!",
                "Sad": "Feeling really down today. Nothing seems to matter and everything feels pointless.",
                "Depressed": "I feel empty inside. Nothing brings me joy anymore. Life has no meaning.",
                "Anxious": "I can't stop worrying about everything. My heart races and I can't sleep.",
                "Neutral": "The weather today is average. I went to the store and bought some groceries."
            }
            
            # Create columns for sample buttons
            sample_cols = st.columns(3)
            sample_items = list(sample_texts.items())
            for idx, (label, text) in enumerate(sample_items):
                with sample_cols[idx % 3]:
                    if st.button(label, use_container_width=True):
                        st.session_state.sample_text = text
                        st.rerun()
            
            # Main text input
            input_text = st.text_area(
                "Type or paste your text here:",
                value=st.session_state.get('sample_text', ''),
                height=200,
                placeholder="Share your thoughts, feelings, or experiences...",
                key="main_input"
            )
            
            col_analyze, col_clear = st.columns([3, 1])
            with col_analyze:
                analyze_clicked = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
            with col_clear:
                if st.button("📋 Clear", use_container_width=True):
                    if 'sample_text' in st.session_state:
                        del st.session_state.sample_text
                    st.rerun()
        
        with col2:
            st.markdown("### 🎯 Live Preview")
            
            if models is not None and input_text.strip():
                # Quick preview analysis
                with st.spinner("Analyzing..."):
                    try:
                        emotion_model, emotion_vectorizer, emotion_le, depression_model, depression_vectorizer = models
                        
                        # Emotion prediction
                        emotion_label, emotion_probs, emotion_conf = predict_emotion(
                            input_text, emotion_model, emotion_vectorizer, emotion_le
                        )
                        
                        # Depression prediction
                        dep_pred, dep_probs, dep_conf = predict_depression(
                            input_text, depression_model, depression_vectorizer
                        )
                        
                        # Display preview
                        icon = emotion_icons.get(emotion_label, '')
                        color = emotion_colors.get(emotion_label, '#000000')
                        
                        st.markdown(f"""
                            <div class='emotion-card'>
                                <div style='font-size: 3rem; text-align: center;'>{icon}</div>
                                <div style='font-size: 1.8rem; font-weight: bold; text-align: center; color: {color}; margin: 1rem 0;'>
                                    {emotion_label.title()}
                                </div>
                                <div style='font-size: 1rem; color: #6B7280; text-align: center;'>
                                    Predicted Emotion
                                </div>
                                <div style='height: 10px; background: #e2e8f0; border-radius: 5px; margin: 1rem 0;'>
                                    <div style='height: 100%; width: {emotion_conf*100}%; background: {color}; border-radius: 5px;'></div>
                                </div>
                                <div style='font-size: 1.2rem; font-weight: bold; text-align: center;'>
                                    {emotion_conf:.1%} confidence
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Depression preview
                        dep_color = "#EF4444" if dep_pred == 1 else "#10B981"
                        dep_icon = "⚠️" if dep_pred == 1 else "✅"
                        dep_text = "Depression Detected" if dep_pred == 1 else "No Depression"
                        
                        st.markdown(f"""
                            <div class='emotion-card' style='border-left: 4px solid {dep_color};'>
                                <div style='font-size: 2rem; text-align: center;'>{dep_icon}</div>
                                <div style='font-size: 1.3rem; font-weight: bold; text-align: center; color: {dep_color}; margin: 0.5rem 0;'>
                                    {dep_text}
                                </div>
                                <div style='font-size: 1rem; color: #6B7280; text-align: center;'>
                                    Depression Analysis
                                </div>
                                <div style='height: 8px; background: #e2e8f0; border-radius: 5px; margin: 1rem 0;'>
                                    <div style='height: 100%; width: {dep_conf*100}%; background: {dep_color}; border-radius: 5px;'></div>
                                </div>
                                <div style='font-size: 1.1rem; font-weight: bold; text-align: center;'>
                                    {dep_conf:.1%} confidence
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Preview error: {str(e)}")
            
            elif not input_text.strip():
                st.info("👆 Enter text above to see live preview")
            
            # Model status
            st.markdown("---")
            st.markdown("### 🔧 Model Status")
            if models is not None:
                st.success("✅ Models ready for analysis")
                emotion_model, emotion_vectorizer, emotion_le, _, _ = models
                emotions_count = len(emotion_le.classes_)
                st.caption(f"Detecting {emotions_count} emotion categories + Depression")
            else:
                st.error("⚠️ Models not loaded")
                st.caption("Please check model files: emotion_model.pkl, depression_model.pkl, etc.")
    
    # ====================== PERFORM ANALYSIS ======================
    if analyze_clicked and input_text.strip() and models is not None:
        with st.spinner('🔬 Analyzing text patterns...'):
            result = analyze_text(input_text, models)
            
            if result:
                # Add to session history
                st.session_state.analysis_history.append(result)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Comprehensive Analysis Results")
                
                # Dual results row
                col1, col2 = st.columns(2)
                
                with col1:
                    # Emotion result
                    icon = emotion_icons.get(result['primary_emotion'], '')
                    color = emotion_colors.get(result['primary_emotion'], '#000000')
                    
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 3rem; text-align: center;'>{icon}</div>
                            <div style='font-size: 1.8rem; font-weight: bold; text-align: center; color: {color}; margin: 1rem 0;'>
                                {result['primary_emotion'].title()}
                            </div>
                            <div style='font-size: 1rem; color: #6B7280; text-align: center;'>
                                Primary Emotion
                            </div>
                            <div style='font-size: 2rem; font-weight: bold; text-align: center; margin-top: 1rem;'>
                                {result['primary_probability']:.1%}
                            </div>
                            <div style='font-size: 0.9rem; color: #6B7280; text-align: center;'>
                                Confidence
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Depression result
                    dep_icon = result['depression_icon']
                    dep_color = result['depression_color']
                    dep_status = result['depression_status']
                    
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 3rem; text-align: center;'>{dep_icon}</div>
                            <div style='font-size: 1.8rem; font-weight: bold; text-align: center; color: {dep_color}; margin: 1rem 0;'>
                                {dep_status}
                            </div>
                            <div style='font-size: 1rem; color: #6B7280; text-align: center;'>
                                Depression Status
                            </div>
                            <div style='font-size: 2rem; font-weight: bold; text-align: center; margin-top: 1rem;'>
                                {result['depression_confidence']:.1%}
                            </div>
                            <div style='font-size: 0.9rem; color: #6B7280; text-align: center;'>
                                Confidence
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Additional metrics
                col3, col4 = st.columns(2)
                with col3:
                    intensity_level = "High" if result['emotional_intensity'] > 0.7 else "Medium" if result['emotional_intensity'] > 0.4 else "Low"
                    st.metric("Emotional Intensity", intensity_level, f"{result['emotional_intensity']:.1%}")
                
                with col4:
                    st.metric("Emotional Complexity", result['emotional_complexity'])
                
                # Emotion distribution chart
                st.markdown("### 📈 Emotion Probability Distribution")
                
                emotion_probs = result['emotion_probabilities']
                fig = go.Figure(data=[
                    go.Bar(
                        x=[ep['emotion'].title() for ep in emotion_probs],
                        y=[ep['probability'] for ep in emotion_probs],
                        marker_color=[ep['color'] for ep in emotion_probs],
                        text=[f"{ep['probability']:.1%}" for ep in emotion_probs],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis_title="Emotions",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top emotions in detail
                st.markdown("### 🏆 Top 3 Detected Emotions")
                
                top_n = min(3, len(emotion_probs))
                cols = st.columns(top_n)
                
                for idx, emotion_data in enumerate(emotion_probs[:top_n]):
                    with cols[idx]:
                        st.markdown(f"""
                            <div class='emotion-card'>
                                <div style='font-size: 2.5rem; text-align: center;'>{emotion_data['icon']}</div>
                                <div style='font-size: 1.3rem; font-weight: bold; text-align: center; color: {emotion_data['color']};'>
                                    {emotion_data['emotion'].title()}
                                </div>
                                <div style='font-size: 2rem; font-weight: bold; text-align: center; margin: 1rem 0;'>
                                    {emotion_data['probability']:.1%}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Word Cloud and Insights
                st.markdown("### 🔍 Text Analysis")
                col_cloud, col_insights = st.columns(2)
                
                with col_cloud:
                    st.markdown("#### 📝 Word Cloud")
                    if result['text']:
                        # Simple preprocessing for word cloud
                        text_for_wc = result['text'].lower()
                        text_for_wc = re.sub(r'[^\w\s]', '', text_for_wc)
                        
                        wordcloud = WordCloud(
                            width=600,
                            height=400,
                            background_color='white',
                            colormap='RdYlBu',
                            max_words=100,
                            contour_width=3,
                            contour_color='steelblue'
                        ).generate(text_for_wc)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                
                with col_insights:
                    st.markdown("#### 💡 Insights & Recommendations")
                    
                    insights = []
                    primary = result['primary_emotion']
                    depression = result['depression_status']
                    
                    # Emotion-based insights
                    if primary in ['happy', 'joy']:
                        insights.append("🌈 **Positive Emotions Detected:** You're expressing uplifting feelings.")
                        insights.append("💫 **Tip:** Share this positivity with others to boost collective mood.")
                    elif primary in ['sad', 'sadness']:
                        insights.append("🌧️ **Sadness Detected:** It's okay to feel down sometimes.")
                        insights.append("🤝 **Recommendation:** Consider talking to a friend or writing in a journal.")
                    elif primary == 'fear':
                        insights.append("😰 **Anxiety/Fear Detected:** Your text shows signs of worry.")
                        insights.append("🧘 **Tip:** Practice deep breathing or mindfulness exercises.")
                    elif primary == 'anger':
                        insights.append("🔥 **Anger Detected:** Strong emotions detected in text.")
                        insights.append("🌳 **Recommendation:** Physical activity can help channel this energy.")
                    
                    # Depression-specific insights
                    if depression == "Detected":
                        insights.append("⚠️ **Depression Alert:** Text shows potential depression indicators.")
                        insights.append("📞 **Important:** Consider speaking with a mental health professional.")
                        insights.append("💪 **Support:** Reach out to friends/family or call a helpline.")
                    else:
                        insights.append("✅ **Mental Health:** No depression indicators detected in text.")
                        insights.append("🌟 **Maintenance:** Continue healthy emotional expression.")
                    
                    # Complexity insight
                    if result['emotional_complexity'] == 'Complex':
                        insights.append("🧠 **Complex Emotions:** You're experiencing multiple feelings simultaneously.")
                        insights.append("📖 **Reflection:** This shows emotional depth and self-awareness.")
                    
                    for insight in insights:
                        st.markdown(f"""
                            <div class='info-box'>
                                {insight}
                            </div>
                        """, unsafe_allow_html=True)
                
                # Save button
                if st.button("💾 Save to History", use_container_width=True):
                    st.success("✅ Analysis saved to history!")
                    st.balloons()
    
    # ====================== TAB 2: DASHBOARD ======================
    with tab2:
        st.markdown("## 📈 Analytics Dashboard")
        
        if st.session_state.analysis_history:
            # Convert session history to DataFrame
            session_df = pd.DataFrame(st.session_state.analysis_history)
            
            # Overall statistics
            st.markdown("### 📊 Overview Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", len(session_df))
            with col2:
                avg_emotion_conf = session_df['primary_probability'].mean()
                st.metric("Avg. Emotion Confidence", f"{avg_emotion_conf:.1%}")
            with col3:
                depression_count = (session_df['depression_status'] == 'Detected').sum()
                st.metric("Depression Detections", depression_count)
            with col4:
                most_common = session_df['primary_emotion'].mode()[0] if len(session_df) > 0 else "None"
                st.metric("Most Common Emotion", most_common.title())
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("### 🎭 Emotion Frequency")
                emotion_counts = session_df['primary_emotion'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Count']
                
                fig_bar = px.bar(emotion_counts, x='Emotion', y='Count', 
                               color='Emotion', color_discrete_map=emotion_colors)
                fig_bar.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_chart2:
                st.markdown("### ⚠️ Depression Status")
                depression_counts = session_df['depression_status'].value_counts().reset_index()
                depression_counts.columns = ['Status', 'Count']
                
                fig_pie = px.pie(depression_counts, values='Count', names='Status',
                               color='Status', color_discrete_map={'Detected': '#EF4444', 'Not Detected': '#10B981'})
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # History table
            st.markdown("### 📋 Analysis History")
            display_data = []
            for idx, row in session_df.iterrows():
                display_data.append({
                    'Time': row['timestamp'].strftime("%H:%M"),
                    'Text Preview': (row['text'][:50] + "...") if len(row['text']) > 50 else row['text'],
                    'Emotion': f"{emotion_icons.get(row['primary_emotion'], '')} {row['primary_emotion'].title()}",
                    'Emotion Conf': f"{row['primary_probability']:.1%}",
                    'Depression': row['depression_status'],
                    'Dep Conf': f"{row['depression_confidence']:.1%}"
                })
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Export options
            st.markdown("---")
            st.markdown("### 📤 Export Data")
            
            col_export, col_clear = st.columns(2)
            
            with col_export:
                if st.button("📥 Export Session History", use_container_width=True):
                    export_df = pd.DataFrame(st.session_state.analysis_history)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="emotion_depression_history.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_clear:
                if st.button("🗑️ Clear Session History", type="secondary", use_container_width=True):
                    st.session_state.analysis_history = []
                    st.success("Session history cleared!")
                    st.rerun()
        else:
            st.info("No analysis history yet. Analyze some text to see your dashboard!")
    
    # ====================== TAB 3: ABOUT ======================
    with tab3:
        st.markdown("## 📚 About This App")
        
        col_about1, col_about2 = st.columns(2)
        
        with col_about1:
            st.markdown("### 🧠 Dual Analysis System")
            st.write("""
            This app combines **two powerful machine learning models**:
            
            1. **Emotion Detection Model**
               - Classifies text into multiple emotional categories
               - Provides confidence scores for each emotion
               - Visualizes emotional probability distribution
            
            2. **Depression Detection Model**  
               - Binary classification for depression risk
               - Analyzes linguistic patterns associated with depression
               - Provides risk assessment with confidence levels
            """)
            
            st.markdown("### 🎯 How It Works")
            st.write("""
            The system processes your text through two pipelines:
            
            **Text → Vectorization → Machine Learning → Results**
            
            1. **Text Preprocessing:** Converts text to numerical features
            2. **Feature Extraction:** Uses TF-IDF to capture important words
            3. **Model Prediction:** Trained ML models analyze the features
            4. **Result Generation:** Provides emotion and depression analysis
            """)
        
        with col_about2:
            st.markdown("### ⚡ Key Features")
            st.markdown("""
            - **Dual Analysis:** Emotion + Depression detection
            - **Real-time Processing:** Instant results as you type
            - **Visual Analytics:** Interactive charts and graphs
            - **History Tracking:** Save and review your analyses
            - **Export Functionality:** Download data for further analysis
            - **Insights & Tips:** Personalized recommendations
            - **Beautiful UI:** Modern, user-friendly interface
            """)
            
            st.markdown("### 🔧 Technical Stack")
            st.markdown("""
            - **Frontend:** Streamlit for interactive interface
            - **ML Framework:** scikit-learn models
            - **Visualization:** Plotly, Matplotlib, WordCloud
            - **Data Processing:** Pandas, NumPy
            - **Deployment:** Local/Cloud ready
            """)
            
            st.markdown("### 🏆 Applications")
            st.markdown("""
            - **Mental Health Screening:** Early depression detection
            - **Therapeutic Tools:** Emotional self-awareness
            - **Research:** Psychological and linguistic studies
            - **Customer Support:** Emotional tone analysis
            - **Education:** Emotional intelligence training
            """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Disclaimer")
        st.warning("""
        **This tool is for informational and educational purposes only.**
        
        - **NOT** a substitute for professional medical advice
        - **NOT** a diagnostic tool
        - **NOT** a replacement for mental health professionals
        
        If you're experiencing emotional distress or mental health concerns, 
        please seek help from qualified healthcare professionals.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #4a5568; font-size: 0.9rem; padding: 1rem 0;'>
            <p><strong>Emotion & Depression Analyzer</strong> | Built with ❤️ using Streamlit</p>
            <p>For educational and research purposes only</p>
        </div>
        """, unsafe_allow_html=True)

# ====================== RUN APP ======================
if __name__ == '__main__':
    main()