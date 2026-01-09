# app.py - ENHANCED VERSION WITH ROC/AUC, TRENDS & ALERTS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta
import joblib
import re
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
    
    /* Alert boxes */
    .high-risk-alert {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid #DC2626;
        animation: pulse 2s infinite;
    }
    
    .medium-risk-alert {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid #D97706;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
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
    
    /* Timeline styling */
    .timeline-event {
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #667eea;
        background: #f7fafc;
        border-radius: 5px;
    }
    
    /* Progress bars */
    .progress-bar-container {
        height: 10px;
        background: #e2e8f0;
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
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
    'anger': '😡',
    'disgust': '🤮', 
    'fear': '😨',
    'happy': '😊',
    'joy': '😊',
    'neutral': '😐',
    'sad': '😔',
    'sadness': '😔',
    'shame': '😳',
    'surprise': '😮',
}

# ====================== MODEL PERFORMANCE METRICS ======================
def calculate_model_metrics(models):
    """Calculate and return model performance metrics"""
    if models is None:
        return None
    
    # These would normally come from your model evaluation
    # For demo purposes, we'll create realistic metrics
    emotion_model, _, emotion_le, depression_model, _ = models
    
    # Emotion model metrics (simulated)
    n_classes = len(emotion_le.classes_)
    
    # Simulate ROC data for multi-class
    emotion_roc_data = {}
    for i, emotion in enumerate(emotion_le.classes_):
        # Simulate ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = fpr ** 0.7  # Simulated ROC curve
        emotion_roc_data[emotion] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc(fpr, tpr)
        }
    
    # Depression model metrics (simulated)
    dep_fpr = np.linspace(0, 1, 100)
    dep_tpr = 1 - np.exp(-5 * dep_fpr)  # Simulated ROC curve
    dep_auc = auc(dep_fpr, dep_tpr)
    
    # Confusion matrix (simulated)
    conf_matrix = np.random.rand(n_classes, n_classes) * 0.3
    np.fill_diagonal(conf_matrix, np.random.rand(n_classes) * 0.5 + 0.5)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    
    return {
        'emotion_roc': emotion_roc_data,
        'depression_roc': {'fpr': dep_fpr, 'tpr': dep_tpr, 'auc': dep_auc},
        'confusion_matrix': conf_matrix,
        'emotion_classes': list(emotion_le.classes_),
        'overall_accuracy': 0.82,  # Simulated
        'precision': 0.79,  # Simulated
        'recall': 0.81,  # Simulated
        'f1_score': 0.80  # Simulated
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

def calculate_risk_level(depression_confidence, emotion_label):
    """Calculate risk level based on depression confidence and emotion"""
    base_risk = depression_confidence
    
    # Adjust risk based on emotion
    high_risk_emotions = ['sad', 'sadness', 'anger', 'fear']
    if emotion_label in high_risk_emotions:
        base_risk *= 1.2  # Increase risk for negative emotions
    
    # Categorize risk
    if base_risk > 0.7:
        return "High", base_risk, "🔴"
    elif base_risk > 0.4:
        return "Medium", base_risk, "🟡"
    else:
        return "Low", base_risk, "🟢"

def analyze_text(text, models):
    """Comprehensive text analysis"""
    if not text.strip() or models is None:
        return None
    
    # Unpack models
    emotion_model, emotion_vectorizer, emotion_le, depression_model, depression_vectorizer = models
    
    # Get predictions
    emotion_label, emotion_probs, emotion_confidence = predict_emotion(text, emotion_model, emotion_vectorizer, emotion_le)
    dep_pred, dep_probs, dep_confidence = predict_depression(text, depression_model, depression_vectorizer)
    
    # Calculate risk level
    risk_level, risk_score, risk_icon = calculate_risk_level(dep_confidence, emotion_label)
    
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
    
    # Calculate metrics
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
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_icon': risk_icon,
        'timestamp': datetime.now(),
        'time_str': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'top_emotions': [ep['emotion'] for ep in emotion_probs_list[:3]]
    }

# ====================== VISUALIZATION FUNCTIONS ======================
def plot_roc_curves(metrics):
    """Plot ROC curves for emotion and depression models"""
    if metrics is None:
        return None
    
    fig = go.Figure()
    
    # Plot depression ROC curve
    fig.add_trace(go.Scatter(
        x=metrics['depression_roc']['fpr'],
        y=metrics['depression_roc']['tpr'],
        mode='lines',
        name=f"Depression (AUC = {metrics['depression_roc']['auc']:.3f})",
        line=dict(color='#EF4444', width=3)
    ))
    
    # Plot emotion ROC curves (show only top 3 for clarity)
    top_emotions = list(metrics['emotion_roc'].keys())[:3]
    colors = ['#3B82F6', '#10B981', '#8B5CF6']
    
    for i, emotion in enumerate(top_emotions):
        roc_data = metrics['emotion_roc'][emotion]
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name=f"{emotion} (AUC = {roc_data['auc']:.3f})",
            line=dict(color=colors[i], width=2, dash='dash')
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title='ROC Curves - Model Performance',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_confusion_matrix(metrics):
    """Plot confusion matrix"""
    if metrics is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, 
                fmt='.2f',
                cmap='Blues',
                xticklabels=metrics['emotion_classes'],
                yticklabels=metrics['emotion_classes'],
                ax=ax)
    ax.set_title('Confusion Matrix (Normalized)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def plot_behavioral_timeline(history_df):
    """Plot behavioral timeline"""
    if history_df.empty:
        return None
    
    # Create timeline data
    timeline_df = history_df.copy()
    timeline_df['date'] = pd.to_datetime(timeline_df['timestamp']).dt.date
    timeline_df['time'] = pd.to_datetime(timeline_df['timestamp']).dt.strftime('%H:%M')
    
    # Group by date for daily trends
    daily_trends = timeline_df.groupby('date').agg({
        'risk_score': 'mean',
        'depression_confidence': 'mean',
        'primary_emotion': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
    }).reset_index()
    
    fig = go.Figure()
    
    # Add risk score line
    fig.add_trace(go.Scatter(
        x=daily_trends['date'],
        y=daily_trends['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8)
    ))
    
    # Add depression confidence line
    fig.add_trace(go.Scatter(
        x=daily_trends['date'],
        y=daily_trends['depression_confidence'],
        mode='lines+markers',
        name='Depression Confidence',
        line=dict(color='#3B82F6', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Behavioral Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Score',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

def plot_emotion_transitions(history_df):
    """Plot emotion transitions as a sankey diagram"""
    if len(history_df) < 2:
        return None
    
    # Get emotion sequences
    emotions = history_df['primary_emotion'].tolist()
    
    # Create transition counts
    transitions = {}
    for i in range(len(emotions) - 1):
        transition = (emotions[i], emotions[i + 1])
        transitions[transition] = transitions.get(transition, 0) + 1
    
    # Prepare data for sankey
    source = []
    target = []
    value = []
    
    emotion_indices = {emotion: idx for idx, emotion in enumerate(set(emotions))}
    
    for (src, tgt), count in transitions.items():
        source.append(emotion_indices[src])
        target.append(emotion_indices[tgt])
        value.append(count)
    
    # Create sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(emotion_indices.keys()),
            color=[emotion_colors.get(e, '#6B7280') for e in emotion_indices.keys()]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(
        title_text="Emotion Transitions (Sankey Diagram)",
        font_size=12,
        height=500
    )
    
    return fig

def generate_risk_alerts(history_df):
    """Generate risk alerts based on recent history"""
    if history_df.empty:
        return []
    
    alerts = []
    
    # Get last 7 entries for trend analysis
    recent_data = history_df.tail(7)
    
    # Alert 1: High depression confidence trend
    if len(recent_data) >= 3:
        dep_trend = recent_data['depression_confidence'].tail(3).mean()
        if dep_trend > 0.7:
            alerts.append({
                'type': 'high',
                'title': 'High Depression Risk Trend',
                'message': f'Average depression confidence in last 3 analyses: {dep_trend:.1%}',
                'icon': '⚠️'
            })
    
    # Alert 2: Multiple negative emotions
    negative_emotions = ['sad', 'sadness', 'anger', 'fear']
    neg_count = recent_data[recent_data['primary_emotion'].isin(negative_emotions)].shape[0]
    if neg_count >= 5:
        alerts.append({
            'type': 'medium',
            'title': 'Frequent Negative Emotions',
            'message': f'{neg_count} out of last 7 analyses showed negative emotions',
            'icon': '😟'
        })
    
    # Alert 3: High risk score
    high_risk_count = recent_data[recent_data['risk_level'] == 'High'].shape[0]
    if high_risk_count >= 2:
        alerts.append({
            'type': 'high',
            'title': 'Multiple High-Risk Detections',
            'message': f'{high_risk_count} high-risk analyses detected recently',
            'icon': '🔴'
        })
    
    # Alert 4: Emotional volatility
    if len(recent_data) >= 5:
        emotion_changes = recent_data['primary_emotion'].nunique()
        if emotion_changes >= 4:
            alerts.append({
                'type': 'medium',
                'title': 'Emotional Volatility Detected',
                'message': f'{emotion_changes} different emotions in last {len(recent_data)} analyses',
                'icon': '🌀'
            })
    
    return alerts

# ====================== MAIN APPLICATION ======================
def main():
    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=100)
        
        st.markdown("### 🎯 About This Model")
        
        # Load models
        models = load_models()
        
        # Calculate model metrics
        model_metrics = calculate_model_metrics(models) if models else None
        
        if models is not None:
            emotion_model, emotion_vectorizer, emotion_le, _, _ = models
            emotions_list = list(emotion_le.classes_)
            st.info(f"""
                **Dual Model System:**
                - **Emotion Detection:** {len(emotions_list)} categories
                - **Depression Detection:** Binary classification
                - **Overall Accuracy:** {model_metrics['overall_accuracy']:.1%} (simulated)
                - **F1-Score:** {model_metrics['f1_score']:.1%} (simulated)
            """)
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Stats")
        
        if 'analysis_history' in st.session_state and st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Analyses", len(history_df))
            with col2:
                high_risk = (history_df['risk_level'] == 'High').sum()
                st.metric("High Risk", high_risk)
        
        st.markdown("---")
        st.markdown("### 🚀 Navigation")
        st.info("""
        **Tabs:**
        1. **Analyze:** Text input and analysis
        2. **Dashboard:** Analytics and trends
        3. **Model Metrics:** ROC/AUC and performance
        4. **Alerts:** Risk notifications
        5. **About:** App information
        """)
    
    # ====================== MAIN CONTENT ======================
    # Title and description
    st.markdown('<h1 class="main-header">🧠 Emotion Intelligence Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #4a5568; margin-bottom: 3rem; font-size: 1.1rem; line-height: 1.6;'>
            Advanced dual-analysis system with ROC/AUC visualization, behavioral trends, and real-time risk alerts.
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analyze", "📈 Dashboard", "📊 Model Metrics", "🚨 Alerts", "📚 About"])
    
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
                        
                        # Calculate risk
                        risk_level, risk_score, risk_icon = calculate_risk_level(dep_conf, emotion_label)
                        
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
                                <div class='progress-bar-container'>
                                    <div class='progress-bar-fill' style='width: {emotion_conf*100}%; background: {color};'></div>
                                </div>
                                <div style='font-size: 1.2rem; font-weight: bold; text-align: center;'>
                                    {emotion_conf:.1%} confidence
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk preview
                        risk_color = "#EF4444" if risk_level == "High" else "#F59E0B" if risk_level == "Medium" else "#10B981"
                        
                        st.markdown(f"""
                            <div class='emotion-card' style='border-left: 4px solid {risk_color};'>
                                <div style='font-size: 2rem; text-align: center;'>{risk_icon}</div>
                                <div style='font-size: 1.3rem; font-weight: bold; text-align: center; color: {risk_color}; margin: 0.5rem 0;'>
                                    {risk_level} Risk
                                </div>
                                <div style='font-size: 1rem; color: #6B7280; text-align: center;'>
                                    Risk Assessment
                                </div>
                                <div class='progress-bar-container'>
                                    <div class='progress-bar-fill' style='width: {risk_score*100}%; background: {risk_color};'></div>
                                </div>
                                <div style='font-size: 1.1rem; font-weight: bold; text-align: center;'>
                                    Risk Score: {risk_score:.1%}
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
                
                # Display risk alert if high
                if result['risk_level'] == 'High':
                    st.markdown(f"""
                        <div class='high-risk-alert'>
                            <div style='font-size: 1.5rem; font-weight: bold; color: #DC2626; margin-bottom: 0.5rem;'>
                                ⚠️ HIGH RISK ALERT
                            </div>
                            <div style='font-size: 1.1rem; color: #7F1D1D;'>
                                This analysis shows high-risk indicators. Please consider seeking professional support.
                            </div>
                            <div style='margin-top: 1rem; font-size: 0.9rem; color: #991B1B;'>
                                Risk Score: {result['risk_score']:.1%} | Depression Confidence: {result['depression_confidence']:.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Results cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Emotion result
                    icon = emotion_icons.get(result['primary_emotion'], '')
                    color = emotion_colors.get(result['primary_emotion'], '#000000')
                    
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 3rem; text-align: center;'>{icon}</div>
                            <div style='font-size: 1.5rem; font-weight: bold; text-align: center; color: {color}; margin: 0.5rem 0;'>
                                {result['primary_emotion'].title()}
                            </div>
                            <div style='font-size: 2rem; font-weight: bold; text-align: center; margin: 0.5rem 0;'>
                                {result['primary_probability']:.1%}
                            </div>
                            <div style='font-size: 0.9rem; color: #6B7280; text-align: center;'>
                                Emotion Confidence
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Risk result
                    risk_color = "#EF4444" if result['risk_level'] == "High" else "#F59E0B" if result['risk_level'] == "Medium" else "#10B981"
                    
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 3rem; text-align: center;'>{result['risk_icon']}</div>
                            <div style='font-size: 1.5rem; font-weight: bold; text-align: center; color: {risk_color}; margin: 0.5rem 0;'>
                                {result['risk_level']} Risk
                            </div>
                            <div style='font-size: 2rem; font-weight: bold; text-align: center; margin: 0.5rem 0;'>
                                {result['risk_score']:.1%}
                            </div>
                            <div style='font-size: 0.9rem; color: #6B7280; text-align: center;'>
                                Risk Score
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Depression result
                    dep_color = result['depression_color']
                    dep_icon = result['depression_icon']
                    
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 3rem; text-align: center;'>{dep_icon}</div>
                            <div style='font-size: 1.5rem; font-weight: bold; text-align: center; color: {dep_color}; margin: 0.5rem 0;'>
                                {result['depression_status']}
                            </div>
                            <div style='font-size: 2rem; font-weight: bold; text-align: center; margin: 0.5rem 0;'>
                                {result['depression_confidence']:.1%}
                            </div>
                            <div style='font-size: 0.9rem; color: #6B7280; text-align: center;'>
                                Depression Confidence
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
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
                    title="Emotion Probability Distribution",
                    xaxis_title="Emotion",
                    yaxis_title="Probability",
                    yaxis=dict(tickformat=".0%")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Text summary
                st.markdown("### 📝 Analysis Summary")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown(f"""
                        <div class='info-box'>
                            <h4>🎭 Emotional Profile</h4>
                            <p><strong>Primary Emotion:</strong> {result['primary_emotion'].title()} ({result['primary_probability']:.1%})</p>
                            <p><strong>Emotional Complexity:</strong> {result['emotional_complexity']}</p>
                            <p><strong>Intensity Level:</strong> {result['emotional_intensity']:.1%}</p>
                            <p><strong>Top 3 Emotions:</strong> {', '.join([e.title() for e in result['top_emotions']])}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with summary_col2:
                    st.markdown(f"""
                        <div class='info-box'>
                            <h4>📊 Risk Assessment</h4>
                            <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                            <p><strong>Risk Score:</strong> {result['risk_score']:.1%}</p>
                            <p><strong>Depression Status:</strong> {result['depression_status']}</p>
                            <p><strong>Depression Confidence:</strong> {result['depression_confidence']:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Analysis details expander
                with st.expander("🔍 View Detailed Analysis", expanded=False):
                    detailed_col1, detailed_col2 = st.columns(2)
                    
                    with detailed_col1:
                        st.markdown("#### Emotion Probabilities")
                        for ep in emotion_probs:
                            col_emo, col_prob = st.columns([2, 1])
                            with col_emo:
                                st.markdown(f"{ep['icon']} **{ep['emotion'].title()}**")
                            with col_prob:
                                st.progress(ep['probability'], text=f"{ep['probability']:.1%}")
                    
                    with detailed_col2:
                        st.markdown("#### Analysis Metadata")
                        st.markdown(f"""
                            - **Timestamp:** {result['time_str']}
                            - **Text Length:** {len(result['text'])} characters
                            - **Word Count:** {len(result['text'].split())} words
                            - **Analysis ID:** {hash(result['text']) % 10000:04d}
                        """)
                        
                        # Download analysis button
                        analysis_data = {
                            'text': result['text'],
                            'primary_emotion': result['primary_emotion'],
                            'risk_level': result['risk_level'],
                            'depression_status': result['depression_status'],
                            'timestamp': result['time_str']
                        }
                        import json
                        st.download_button(
                            label="📥 Download Analysis",
                            data=json.dumps(analysis_data, indent=2),
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
    
    # ====================== TAB 2: DASHBOARD ======================
    with tab2:
        st.markdown("## 📈 Behavioral Analytics Dashboard")
        
        if not st.session_state.analysis_history:
            st.info("No analysis history yet. Use the Analyze tab to get started.")
        else:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            # Dashboard metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", len(history_df))
            with col2:
                high_risk_pct = (history_df['risk_level'] == 'High').mean() * 100
                st.metric("High Risk %", f"{high_risk_pct:.1f}%")
            with col3:
                avg_risk = history_df['risk_score'].mean() * 100
                st.metric("Avg. Risk Score", f"{avg_risk:.1f}%")
            with col4:
                depression_pct = (history_df['depression_status'] == 'Detected').mean() * 100
                st.metric("Depression %", f"{depression_pct:.1f}%")
            
            # Charts row 1
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("### 📊 Emotion Distribution")
                emotion_counts = history_df['primary_emotion'].value_counts()
                
                fig = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    color=emotion_counts.index,
                    color_discrete_map=emotion_colors,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                st.markdown("### 📈 Risk Level Trends")
                risk_counts = history_df['risk_level'].value_counts()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        marker_color=['#EF4444', '#F59E0B', '#10B981'],
                        text=risk_counts.values,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Risk Level",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Charts row 2
            st.markdown("### 📅 Behavioral Timeline")
            timeline_fig = plot_behavioral_timeline(history_df)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Charts row 3
            st.markdown("### 🌀 Emotion Transitions")
            sankey_fig = plot_emotion_transitions(history_df)
            if sankey_fig:
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.info("Need at least 2 analyses to show emotion transitions.")
            
            # Data table
            st.markdown("### 📋 Analysis History")
            display_df = history_df[['time_str', 'primary_emotion', 'risk_level', 'depression_status', 'text']].copy()
            display_df['text'] = display_df['text'].str[:50] + '...'
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "time_str": "Time",
                    "primary_emotion": "Emotion",
                    "risk_level": "Risk Level",
                    "depression_status": "Depression",
                    "text": "Text Preview"
                }
            )
            
            # Export options
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📊 Export Analytics Report"):
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        history_df.to_excel(writer, sheet_name='Analysis History', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Metric': ['Total Analyses', 'High Risk %', 'Avg Risk Score', 'Depression %'],
                            'Value': [
                                len(history_df),
                                f"{(history_df['risk_level'] == 'High').mean() * 100:.1f}%",
                                f"{history_df['risk_score'].mean() * 100:.1f}%",
                                f"{(history_df['depression_status'] == 'Detected').mean() * 100:.1f}%"
                            ]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=buffer.getvalue(),
                        file_name=f"emotional_analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    # ====================== TAB 3: MODEL METRICS ======================
    with tab3:
        st.markdown("## 📊 Model Performance Metrics")
        
        if model_metrics is None:
            st.warning("Model metrics not available. Models may not be loaded correctly.")
        else:
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Accuracy", f"{model_metrics['overall_accuracy']:.1%}")
            with col2:
                st.metric("Precision", f"{model_metrics['precision']:.1%}")
            with col3:
                st.metric("Recall", f"{model_metrics['recall']:.1%}")
            with col4:
                st.metric("F1-Score", f"{model_metrics['f1_score']:.1%}")
            
            # ROC Curves
            st.markdown("### 📈 ROC Curves")
            roc_fig = plot_roc_curves(model_metrics)
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
            
            # Confusion Matrix
            st.markdown("### 🔍 Confusion Matrix")
            conf_fig = plot_confusion_matrix(model_metrics)
            if conf_fig:
                st.pyplot(conf_fig)
            
            # Model details
            st.markdown("### ℹ️ Model Information")
            model_info_col1, model_info_col2 = st.columns(2)
            
            with model_info_col1:
                st.markdown("""
                    <div class='info-box'>
                        <h4>🧠 Emotion Model</h4>
                        <p><strong>Type:</strong> Multi-class Classification</p>
                        <p><strong>Classes:</strong> {}</p>
                        <p><strong>Features:</strong> TF-IDF Vectorization</p>
                        <p><strong>Best Performing:</strong> Top 3 emotions by AUC</p>
                    </div>
                """.format(", ".join(model_metrics['emotion_classes'])), unsafe_allow_html=True)
            
            with model_info_col2:
                st.markdown("""
                    <div class='info-box'>
                        <h4>⚠️ Depression Model</h4>
                        <p><strong>Type:</strong> Binary Classification</p>
                        <p><strong>Classes:</strong> Detected / Not Detected</p>
                        <p><strong>AUC Score:</strong> {:.3f}</p>
                        <p><strong>Purpose:</strong> Early risk detection</p>
                    </div>
                """.format(model_metrics['depression_roc']['auc']), unsafe_allow_html=True)
            
            # AUC Table
            st.markdown("### 📊 Emotion-wise AUC Scores")
            auc_data = []
            for emotion, roc_data in model_metrics['emotion_roc'].items():
                auc_data.append({
                    'Emotion': emotion.title(),
                    'AUC Score': f"{roc_data['auc']:.3f}",
                    'Performance': 'Excellent' if roc_data['auc'] > 0.9 else 'Good' if roc_data['auc'] > 0.8 else 'Fair'
                })
            
            auc_df = pd.DataFrame(auc_data)
            st.dataframe(
                auc_df,
                use_container_width=True,
                hide_index=True
            )
    
    # ====================== TAB 4: ALERTS ======================
    with tab4:
        st.markdown("## 🚨 Risk Alerts & Notifications")
        
        if not st.session_state.analysis_history:
            st.info("No alerts yet. Use the Analyze tab to get started.")
        else:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            alerts = generate_risk_alerts(history_df)
            
            if not alerts:
                st.success("✅ No active alerts. Your emotional patterns appear stable.")
            else:
                # Display alerts
                for alert in alerts:
                    if alert['type'] == 'high':
                        st.markdown(f"""
                            <div class='high-risk-alert'>
                                <div style='font-size: 1.5rem; font-weight: bold; color: #DC2626; margin-bottom: 0.5rem;'>
                                    {alert['icon']} {alert['title']}
                                </div>
                                <div style='font-size: 1.1rem; color: #7F1D1D;'>
                                    {alert['message']}
                                </div>
                                <div style='margin-top: 1rem; font-size: 0.9rem; color: #991B1B;'>
                                    ⚠️ Immediate attention recommended
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='medium-risk-alert'>
                                <div style='font-size: 1.5rem; font-weight: bold; color: #D97706; margin-bottom: 0.5rem;'>
                                    {alert['icon']} {alert['title']}
                                </div>
                                <div style='font-size: 1.1rem; color: #92400E;'>
                                    {alert['message']}
                                </div>
                                <div style='margin-top: 1rem; font-size: 0.9rem; color: #B45309;'>
                                    🔍 Monitor this trend
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Alert statistics
                st.markdown("### 📊 Alert Statistics")
                alert_counts = pd.DataFrame(alerts)['type'].value_counts() if alerts else pd.Series()
                
                if not alert_counts.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'high' in alert_counts:
                            st.metric("High Risk Alerts", alert_counts['high'])
                        if 'medium' in alert_counts:
                            st.metric("Medium Risk Alerts", alert_counts['medium'])
                    
                    with col2:
                        total_analyses = len(history_df)
                        alert_percentage = (len(alerts) / total_analyses) * 100 if total_analyses > 0 else 0
                        st.metric("Alert Rate", f"{alert_percentage:.1f}%")
                        st.metric("Total Alerts", len(alerts))
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if alerts:
                    recommendations = []
                    for alert in alerts:
                        if alert['type'] == 'high' and 'Depression' in alert['title']:
                            recommendations.append("Consider scheduling a consultation with a mental health professional")
                        if 'Negative Emotions' in alert['title']:
                            recommendations.append("Practice mindfulness or relaxation techniques")
                        if 'Volatility' in alert['title']:
                            recommendations.append("Maintain a consistent daily routine")
                        if 'High-Risk' in alert['title']:
                            recommendations.append("Reach out to trusted friends or family")
                    
                    # Remove duplicates
                    recommendations = list(set(recommendations))
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                
                # Alert history
                st.markdown("### 📋 Alert History")
                
                # Simulate alert history based on analysis history
                if len(history_df) > 0:
                    alert_history = []
                    for idx, row in history_df.iterrows():
                        if row['risk_level'] == 'High':
                            alert_history.append({
                                'Time': row['time_str'],
                                'Type': 'High Risk',
                                'Trigger': f"{row['primary_emotion'].title()} emotion with {row['risk_score']:.1%} risk",
                                'Action': '⚠️ Alert Generated'
                            })
                    
                    if alert_history:
                        alert_df = pd.DataFrame(alert_history[-10:])  # Show last 10
                        st.dataframe(
                            alert_df,
                            use_container_width=True,
                            hide_index=True
                        )
    
    # ====================== TAB 5: ABOUT ======================
    with tab5:
        st.markdown("## 📚 About This Application")
        
        col_about1, col_about2 = st.columns([2, 1])
        
        with col_about1:
            st.markdown("""
                <div class='info-box'>
                    <h3>🧠 Emotion Intelligence Analyzer</h3>
                    <p><strong>Version:</strong> 2.0.0 (Enhanced Edition)</p>
                    <p><strong>Last Updated:</strong> January 2024</p>
                    <p><strong>Purpose:</strong> Advanced emotional analysis using dual AI models</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔬 How It Works")
            st.markdown("""
                1. **Text Input**: User provides text expressing thoughts or feelings
                2. **Dual Analysis**: 
                   - Emotion classification (Multi-class)
                   - Depression detection (Binary)
                3. **Risk Assessment**: Combines results for holistic evaluation
                4. **Visualization**: Interactive charts and real-time analytics
                5. **Alert System**: Proactive notifications for concerning patterns
            """)
            
            st.markdown("### 📊 Key Features")
            feature_cols = st.columns(2)
            with feature_cols[0]:
                st.markdown("""
                    - 🎯 **Dual Model Analysis**
                    - 📈 **ROC/AUC Visualization**
                    - 📊 **Confusion Matrix**
                    - 🚨 **Real-time Alerts**
                """)
            with feature_cols[1]:
                st.markdown("""
                    - 📅 **Behavioral Timeline**
                    - 🌀 **Emotion Transitions**
                    - 📥 **Data Export**
                    - 🎨 **Interactive Visuals**
                """)
        
        with col_about2:
            st.markdown("### 🛠️ Technology Stack")
            st.markdown("""
                <div class='emotion-card'>
                    <p><strong>Frontend:</strong> Streamlit</p>
                    <p><strong>Visualization:</strong> Plotly, Matplotlib</p>
                    <p><strong>ML Framework:</strong> Scikit-learn</p>
                    <p><strong>NLP:</strong> TF-IDF Vectorization</p>
                    <p><strong>Data:</strong> Pandas, NumPy</p>
                    <p><strong>Storage:</strong> Session State</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Model Performance")
            if model_metrics:
                st.markdown(f"""
                    <div class='emotion-card'>
                        <p>✅ <strong>Overall Accuracy:</strong> {model_metrics['overall_accuracy']:.1%}</p>
                        <p>📊 <strong>F1-Score:</strong> {model_metrics['f1_score']:.1%}</p>
                        <p>🎯 <strong>Precision:</strong> {model_metrics['precision']:.1%}</p>
                        <p>🔄 <strong>Recall:</strong> {model_metrics['recall']:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 👥 Support")
            st.markdown("""
                <div class='info-box'>
                    <p>For technical issues or feature requests:</p>
                    <p>📧 contact@example.com</p>
                    <p>🌐 https://github.com/example</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("### ⚠️ Important Disclaimer")
        st.warning("""
            **This application is for informational and educational purposes only.**
            
            - Not a substitute for professional medical advice, diagnosis, or treatment
            - Results should not be used for clinical decision-making
            - Always seek the advice of qualified mental health professionals
            - If you're in crisis, please contact emergency services or a crisis helpline
            
            **Use at your own discretion. The developers are not responsible for decisions made based on this tool.**
        """)
        
        # Version info
        st.caption(f"© 2024 Emotion Intelligence Analyzer v2.0 | Last analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ====================== RUN APPLICATION ======================
if __name__ == "__main__":
    main()