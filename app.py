import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="ADHD Screening Tool",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    """Loads the saved model, scaler, and feature columns."""
    try:
        model = joblib.load("adhd_cpu_fixed_model.joblib")
        scaler = joblib.load("scaler.joblib")
        try:
            feature_cols = joblib.load("feature_columns.joblib")
        except FileNotFoundError:
            feature_cols = joblib.load("feature_cols.joblib")
        return model, scaler, feature_cols
    except FileNotFoundError:
        return None, None, None


model, scaler, feature_cols = load_artifacts()

try:
    feature_selector = joblib.load("feature_selector.joblib")
except FileNotFoundError:
    feature_selector = None


def calculate_symptom_score(answers):
    """Calculates a direct, weighted score from the questionnaire (0-100)."""
    weights = {
        'inattention': 1.5,
        'impulsivity': 1.5,
        'sustained_attention': 1.0,
        'hyperactivity': 1.0,
        'consistency': 1.0
    }
    raw_score = sum(answers[key] * weights[key] for key in answers)
    min_score = sum(1 * w for w in weights.values())
    max_score = sum(5 * w for w in weights.values())
    normalized_score = ((raw_score - min_score) / (max_score - min_score)) * 100
    return normalized_score


def get_ai_score(answers, model, scaler, all_features):
    """
    Generates the AI model's prediction and normalizes it to a 0–100 scale.
    Includes feature alignment, selection, and adaptive normalization.
    """
    def map_questionnaire_to_features(answers, all_features):
        ranges = {
            'omissions': (2, 20),
            'commissions': (5, 30),
            'hrt': (350, 550),
            'variability': (15, 60),
            'd_prime': (3.5, 0.5),
            'default': (0, 1)
        }

        def scale_value(score, value_range, reverse=False):
            min_val, max_val = value_range
            if reverse:
                min_val, max_val = max_val, min_val
            return np.interp(score, [1, 5], [min_val, max_val])

        feature_values = {}
        for feature in all_features:
            f_lower = feature.lower()
            if 'omis' in f_lower:
                feature_values[feature] = scale_value(answers['inattention'], ranges['omissions'])
            elif 'commis' in f_lower:
                feature_values[feature] = scale_value(answers['impulsivity'], ranges['commissions'])
            elif 'hrt' in f_lower:
                feature_values[feature] = scale_value(answers['inattention'], ranges['hrt'])
            elif 'var' in f_lower:
                feature_values[feature] = scale_value(answers['consistency'], ranges['variability'])
            elif "d'" in f_lower or 'prime' in f_lower:
                feature_values[feature] = scale_value(answers['consistency'], ranges['d_prime'], reverse=True)
            else:
                feature_values[feature] = np.mean(ranges['default'])
        return feature_values

    feature_data = map_questionnaire_to_features(answers, all_features)
    input_df = pd.DataFrame([feature_data])

    expected_features = getattr(scaler, "feature_names_in_", all_features)
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    scaled_input = scaler.transform(input_df)

    if 'feature_selector' in globals() and feature_selector is not None:
        scaled_input = feature_selector.transform(scaled_input)

    raw_prediction = model.predict(scaled_input)[0]

    AI_MIN_EST = 20
    AI_MAX_EST = 85

    normalized_prediction = np.interp(
        raw_prediction,
        [AI_MIN_EST, AI_MAX_EST],
        [0, 100]
    )

    return np.clip(normalized_prediction, 0, 100)


st.title("🧠 ADHD Screening Tool")

st.warning(
    "**Disclaimer:** This tool assists in identifying potential ADHD-related patterns. "
    "It is for **educational and informational purposes only**. "
    "A formal diagnosis can only be made by a licensed professional."
)

if model is None:
    st.error("**Error:** Model files not found. Please ensure `.joblib` files are in this folder.")
else:
    st.info("Please answer the questions below to help personalize your learning profile.")

    answers = {}
    answers['inattention'] = st.slider("**Inattention:** How often do you make careless mistakes or have trouble paying attention?", 1, 5, 3)
    answers['impulsivity'] = st.slider("**Impulsivity:** How often do you act without thinking or interrupt others?", 1, 5, 3)
    answers['sustained_attention'] = st.slider("**Sustained Attention:** How often do you find it hard to stay focused on long tasks?", 1, 5, 3)
    answers['hyperactivity'] = st.slider("**Hyperactivity/Restlessness:** How often do you fidget or find it difficult to stay seated?", 1, 5, 3)
    answers['consistency'] = st.slider("**Task Consistency:** How much does the quality of your work fluctuate?", 1, 5, 3, help="1=Very Stable, 5=Very Inconsistent")

    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Generate My Learning Profile Score", use_container_width=True):
            symptom_score = calculate_symptom_score(answers)
            ai_score = get_ai_score(answers, model, scaler, feature_cols)

            symptom_weight = 0.70
            ai_weight = 0.30
            final_score = (symptom_score * symptom_weight) + (ai_score * ai_weight)

            st.write("---")
            st.subheader("Your Personalized ADHD Indicator")
            st.metric(label="Score", value=f"{final_score:.0f} / 100")

            if final_score > 70:
                st.error(
                    "**High Indication of ADHD-related Traits:** Strong presence of ADHD-like patterns. "
                    "This does not confirm a diagnosis — consult a mental health professional."
                )
            elif final_score > 40:
                st.warning(
                    "**Moderate Indication:** Some characteristics align with ADHD patterns. "
                    "Monitoring or professional advice may be helpful."
                )
            else:
                st.success(
                    "**Low Indication:** Your responses do not strongly align with ADHD characteristics. "
                    "Still, consult a professional if you have concerns."
                )

    with st.expander("🔍 How is this score calculated?"):
        st.markdown("""
        The final score combines:
        - **70% Direct Score:** Based on your answers — intuitive and self-reported.
        - **30% AI Pattern Analysis:** Uses machine learning to uncover hidden response patterns.
        
        Together, they give a more complete, personalized insight.
        """)
