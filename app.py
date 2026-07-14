import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import requests
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
import io
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Cross-Sell Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.3);
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .predict-box-yes {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .predict-box-no {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .predict-box-yes h2, .predict-box-no h2 {
        color: white;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }
    .predict-box-yes p, .predict-box-no p {
        color: #d1fae5;
        margin: 0;
        font-size: 0.95rem;
    }
    .predict-box-no p { color: #fecaca; }
    .ai-box {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .ai-box h4 { color: #a5b4fc; margin: 0 0 0.8rem 0; }
    .ai-box p { color: #e0e7ff; margin: 0; line-height: 1.6; }
    .section-header {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.5rem 0;
        border-bottom: 2px solid #334155;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1, #075985);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Gemini AI insight ───────────────────────────────────────────────────────────
def get_gemini_insight(customer_data: dict, prediction: int, probability: float,
                       top_features: list, api_key: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}

    feature_text = "\n".join(
        [f"- {feat}: {impact}" for feat, impact in top_features[:5]]
    )

    outcome = "LIKELY to purchase vehicle insurance" if prediction == 1 else "UNLIKELY to purchase vehicle insurance"

    prompt = f"""You are an expert insurance business analyst. Analyze this customer profile model prediction, and SHAP values to provide exactly 3 distinct paragraphs of high-level business insights.

Customer Profile Data:
- Age: {customer_data.get('Age', 'N/A')} years old
- Gender: {customer_data.get('Gender', 'N/A')}
- Vehicle Age Group: {customer_data.get('Vehicle_Age', 'N/A')}
- Has Vehicle Damage History?: {customer_data.get('Vehicle_Damage', 'N/A')}
- Is Already Previously Insured?: {"Yes (1)" if customer_data.get('Previously_Insured') == 1 else "No (0)"}
- Annual Premium: ₹{customer_data.get('Annual_Premium', 'N/A')}
- Customer Vintage: {customer_data.get('Vintage', 'N/A')} days with company
- Current Sales Channel ID Used: {customer_data.get('Policy_Sales_Channel', 'N/A')}

Model Prediction:
The XGBoost model predicts the customer is {outcome} with a confidence of {probability:.1%}.

SHAP Feature Impacts (Important for Interpretation):
{feature_text}
*(Note: Positive impact means it pushes the customer to buy (Lighter/Green). Negative impact means it actively discourages them or drags the probability down (Red).)*

Strict Output Format Instructions (Write exactly 3 paragraphs, NO bullet points, NO headers):
Paragraph 1: Validate the prediction. Explain clearly how the top positive SHAP factors (like vehicle damage or not being insured) and negative factors (like the specific sales channel or vehicle age) interact to justify this prediction.
Paragraph 2: Provide 2-3 highly actionable, concrete strategies for the sales team. If a specific 'Policy_Sales_Channel' has a massive negative impact, suggest shifting to a better channel (e.g., direct agent vs automated SMS) and how to pitch based on their damage history.
Paragraph 3: Identify one major hidden risk or long-term customer relationship opportunity based on their Vintage days and premium size."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.3
        }
    }

    try:
        resp = requests.post(f"{url}?key={api_key}", headers=headers,
                             json=payload, timeout=15)
        data = resp.json()

        # Handle API error response
        if "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown error')}"

        # Extract text from response
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response generated. Please check your API key."

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return "Empty response from Gemini."

        return parts[0].get("text", "No text in response.")

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "Connection error. Please check your internet connection."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# ── Model training ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    data = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))

    feature_cols = ['Age', 'Gender', 'Driving_License', 'Previously_Insured',
                    'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
                    'Policy_Sales_Channel', 'Vintage']
    feature_cols = [c for c in feature_cols if c in data.columns]

    X = data[feature_cols]
    y = data['Response']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, y_pred)
    }

    explainer = shap.TreeExplainer(model)

    return model, explainer, feature_cols, metrics

# ── Preprocess single input ─────────────────────────────────────────────────────
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    gender_map = {'Male': 1, 'Female': 0}
    vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    damage_map = {'Yes': 1, 'No': 0}

    return pd.DataFrame([{
        'Age': input_dict['Age'],
        'Gender': gender_map[input_dict['Gender']],
        'Driving_License': input_dict['Driving_License'],
        'Previously_Insured': input_dict['Previously_Insured'],
        'Vehicle_Age': vehicle_age_map[input_dict['Vehicle_Age']],
        'Vehicle_Damage': damage_map[input_dict['Vehicle_Damage']],
        'Annual_Premium': input_dict['Annual_Premium'],
        'Policy_Sales_Channel': input_dict['Policy_Sales_Channel'],
        'Vintage': input_dict['Vintage'],
    }])

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Insurance Cross-Sell Predictor</h1>
    <p>XGBoost · SHAP Explainability · Gemini AI Insight · Health Insurance → Vehicle Insurance</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    gemini_key = st.text_input("Gemini API Key", type="password",
                               placeholder="AIza...",
                               help="Required for AI insight. Get yours at aistudio.google.com")

    st.markdown("---")
    st.markdown("### 📁 Upload Dataset")
    uploaded = st.file_uploader("Upload train.csv from Kaggle",
                                type=['csv'],
                                help="Download from: kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction")

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b; font-size:0.8rem; line-height:1.6'>
    <b style='color:#94a3b8'>Dataset columns needed:</b><br>
    Age, Gender, Driving_License,<br>
    Previously_Insured, Vehicle_Age,<br>
    Vehicle_Damage, Annual_Premium,<br>
    Policy_Sales_Channel, Vintage,<br>
    Response (target)
    </div>
    """, unsafe_allow_html=True)

# ── Main content ────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Please upload the dataset from the sidebar to get started.")

    st.markdown("### 📖 About This App")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🤖 XGBoost Model**
        Gradient boosting with class imbalance handling, 300 estimators, optimized for AUC-ROC.
        """)
    with col2:
        st.markdown("""
        **🔍 SHAP Explainability**
        Understand *why* the model made each prediction with feature importance waterfall charts.
        """)
    with col3:
        st.markdown("""
        **✨ Gemini AI Insight**
        Get human-readable business recommendations powered by Google Gemini.
        """)
    st.stop()

# ── Load & train ────────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded)

required_cols = ['Age', 'Gender', 'Driving_License', 'Previously_Insured',
                 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
                 'Policy_Sales_Channel', 'Vintage', 'Response']

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

with st.spinner("🔄 Training XGBoost model... (first run may take ~30 seconds)"):
    model, explainer, feature_cols, metrics = train_model(df)

# ── Model metrics ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Model Performance</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Accuracy</div>
        <div class="value">{metrics['accuracy']:.1%}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">ROC-AUC</div>
        <div class="value">{metrics['roc_auc']:.3f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    precision = metrics['report']['1']['precision']
    st.markdown(f"""<div class="metric-card">
        <div class="label">Precision (Class 1)</div>
        <div class="value">{precision:.1%}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    recall = metrics['report']['1']['recall']
    st.markdown(f"""<div class="metric-card">
        <div class="label">Recall (Class 1)</div>
        <div class="value">{recall:.1%}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📈 Feature Importance", "📋 Dataset Overview"])

# ── Tab 1: Single Prediction ────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Customer Profile Input</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        age = st.slider("Age", 18, 85, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
        vehicle_damage = st.selectbox("Vehicle Damage History", ["Yes", "No"])

    with col_b:
        driving_license = st.selectbox("Driving License", [1, 0],
                                       format_func=lambda x: "Has License" if x == 1 else "No License")
        previously_insured = st.selectbox("Previously Insured",
                                          [0, 1],
                                          format_func=lambda x: "Not Insured" if x == 0 else "Already Insured")
        annual_premium = st.number_input("Annual Premium (₹)", 2000, 100000, 30000, step=1000)
        policy_channel = st.number_input("Policy Sales Channel", 1, 200, 152)
        vintage = st.slider("Days with Company (Vintage)", 10, 300, 150)

    if st.button("🔮 Predict & Analyze"):
        customer = {
            'Age': age, 'Gender': gender, 'Driving_License': driving_license,
            'Previously_Insured': previously_insured, 'Vehicle_Age': vehicle_age,
            'Vehicle_Damage': vehicle_damage, 'Annual_Premium': annual_premium,
            'Policy_Sales_Channel': policy_channel, 'Vintage': vintage
        }

        X_input = preprocess_input(customer)[feature_cols]
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        # Prediction result
        if pred == 1:
            st.markdown(f"""<div class="predict-box-yes">
                <h2>✅ Likely to Purchase</h2>
                <p>Confidence: <strong>{prob:.1%}</strong> — This customer is a strong candidate for vehicle insurance cross-sell.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="predict-box-no">
                <h2>❌ Unlikely to Purchase</h2>
                <p>Confidence: <strong>{1-prob:.1%}</strong> — This customer is not likely to purchase vehicle insurance at this time.</p>
            </div>""", unsafe_allow_html=True)

        # SHAP explanation + AI side by side
        st.markdown('<p class="section-header">🔍 Why This Prediction?</p>', unsafe_allow_html=True)

        shap_values = explainer.shap_values(X_input)
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

        feature_shap = list(zip(feature_cols, shap_vals))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        top_n = feature_shap[:7]

        names = [f[0] for f in top_n]
        values = [f[1] for f in top_n]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in values]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], alpha=0.85)
        for bar, val in zip(bars, values[::-1]):
            ax.text(val + 0.01 if val >= 0 else val - 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center',
                    ha='left' if val >= 0 else 'right',
                    color='#94a3b8', fontsize=8)
        ax.axvline(0, color='#475569', linewidth=1)
        ax.set_xlabel('SHAP Value (impact on prediction)', color='#94a3b8', fontsize=9)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.set_title('Feature Impact on This Prediction', color='#e2e8f0', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Top features for AI
        top_features = [(f, f"{'increases' if v > 0 else 'decreases'} likelihood ({abs(v):.3f})")
                        for f, v in feature_shap[:5]]

        # Gemini AI insight
        st.markdown('<p class="section-header">✨ AI Business Insight</p>', unsafe_allow_html=True)
        if gemini_key:
            with st.spinner("Getting Gemini insight..."):
                insight = get_gemini_insight(customer, pred, prob, top_features, gemini_key)
            st.markdown(f"""<div class="ai-box">
                <h4>🤖 Gemini Analysis</h4>
                <p>{insight}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="ai-box">
                <h4>🤖 Gemini Analysis</h4>
                <p style="color:#94a3b8">Add your <strong style="color:#a5b4fc">Gemini API Key</strong> in the sidebar to unlock AI-powered business insights.</p>
            </div>""", unsafe_allow_html=True)

# ── Tab 2: Feature Importance ───────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">📈 Global Feature Importance (XGBoost)</p>',
                unsafe_allow_html=True)

    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    feat_imp = feat_imp.sort_values('Importance', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    fig2.patch.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')

    bars2 = ax2.barh(feat_imp['Feature'], feat_imp['Importance'],
                     color='#38bdf8', alpha=0.8)

    for bar, val in zip(bars2, feat_imp['Importance']):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha='left', color='#94a3b8', fontsize=8)

    ax2.set_xlabel('Feature Importance Score', color='#94a3b8', fontsize=9)
    ax2.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#334155')
    ax2.set_title('Which features matter most for cross-sell prediction?',
                  color='#e2e8f0', fontsize=10, pad=10)

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.markdown("### 📝 Feature Descriptions")
    feat_desc = {
        'Previously_Insured': 'Whether customer already has vehicle insurance — strongest negative predictor',
        'Vehicle_Damage': 'History of vehicle damage — customers with damage more likely to want insurance',
        'Vehicle_Age': 'Age of vehicle — older vehicles correlate with insurance interest',
        'Age': 'Customer age — middle-aged customers tend to be more interested',
        'Policy_Sales_Channel': 'Channel through which customer was contacted',
        'Annual_Premium': 'Amount paid for health insurance premium',
        'Vintage': 'Number of days customer has been with the company',
        'Driving_License': 'Whether customer holds a valid driving license',
        'Gender': 'Customer gender'
    }
    for feat, desc in feat_desc.items():
        if feat in feature_cols:
            st.markdown(f"- **{feat}**: {desc}")

# ── Tab 3: Dataset Overview ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">📋 Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
    with c2:
        interested = df['Response'].sum()
        st.metric("Interested in Vehicle Insurance", f"{interested:,} ({interested/len(df):.1%})")
    with c3:
        st.metric("Features Used", len(feature_cols))

    st.dataframe(df.head(100), use_container_width=True, height=300)

    st.markdown("### Distribution: Target Variable")
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    fig3.patch.set_facecolor('#1e293b')
    ax3.set_facecolor('#1e293b')
    counts = df['Response'].value_counts()
    bars3 = ax3.bar(['Not Interested (0)', 'Interested (1)'], counts.values,
            color=['#ef4444', '#10b981'], alpha=0.8, width=0.3)
    for bar, v in zip(bars3, counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 500,
                 f'{v:,}', ha='center', va='bottom', color='#94a3b8', fontsize=9)
    ax3.set_ylabel('Count', color='#94a3b8', fontsize=9)
    ax3.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#334155')
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()