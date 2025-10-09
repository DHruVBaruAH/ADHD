import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ====================================================================
# Page Configuration
# ====================================================================
st.set_page_config(
    page_title="Personalized Learning Screener",
    page_icon="🧠",
    layout="centered"
)

# ====================================================================
# Load Model Artifacts
# ====================================================================
@st.cache_resource
def load_artifacts():
    """Loads the saved model, scaler, and feature columns."""
    try:
        model = joblib.load("adhd_tuned_model.joblib")
        scaler = joblib.load("scaler.joblib")
        feature_cols = joblib.load("feature_cols.joblib")
        return model, scaler, feature_cols
    except FileNotFoundError:
        return None, None, None

model, scaler, feature_cols = load_artifacts()

# ====================================================================
# Scoring Logic
# ====================================================================

def calculate_symptom_score(answers):
    """
    Calculates a direct, weighted score from the questionnaire (0-100).
    This provides the intuitive part of the final score.
    """
    weights = {'inattention': 1.5, 'impulsivity': 1.5, 'sustained_attention': 1.0, 'hyperactivity': 1.0, 'consistency': 1.0}
    raw_score = sum(answers[key] * weights[key] for key in answers)
    min_score = sum(1 * w for w in weights.values())
    max_score = sum(5 * w for w in weights.values())
    
    # Normalize to 0-100
    normalized_score = ((raw_score - min_score) / (max_score - min_score)) * 100
    return normalized_score

def get_ai_score(answers, model, scaler, all_features):
    """
    Generates the AI model's prediction and normalizes it to a 0-100 scale.
    This provides the pattern-analysis part of the final score.
    """
    # This mapping function translates questionnaire answers to the model's expected technical inputs
    def map_questionnaire_to_features(answers, all_features):
        ranges = {'omissions': (2, 20), 'commissions': (5, 30), 'hrt': (350, 550), 'variability': (15, 60), 'd_prime': (3.5, 0.5), 'default': (0, 1)}
        def scale_value(score, value_range, reverse=False):
            min_val, max_val = value_range
            if reverse: min_val, max_val = max_val, min_val
            return np.interp(score, [1, 5], [min_val, max_val])

        feature_values = {}
        for feature in all_features:
            f_lower = feature.lower()
            if 'omis' in f_lower: feature_values[feature] = scale_value(answers['inattention'], ranges['omissions'])
            elif 'commis' in f_lower: feature_values[feature] = scale_value(answers['impulsivity'], ranges['commissions'])
            elif 'hrt' in f_lower: feature_values[feature] = scale_value(answers['inattention'], ranges['hrt'])
            elif 'var' in f_lower: feature_values[feature] = scale_value(answers['consistency'], ranges['variability'])
            elif 'd\'' in f_lower or 'prime' in f_lower: feature_values[feature] = scale_value(answers['consistency'], ranges['d_prime'], reverse=True)
            else: feature_values[feature] = np.mean(ranges['default'])
        return feature_values

    # 1. Map answers to technical features
    feature_data = map_questionnaire_to_features(answers, all_features)
    input_df = pd.DataFrame([feature_data])
    
    # 2. Scale the features
    scaled_input = scaler.transform(input_df[all_features])

    # 3. Get the model's raw prediction
    raw_prediction = model.predict(scaled_input)[0]

    # 4. Normalize the model's prediction to a 0-100 scale
    # NOTE: These min/max values are assumptions about the model's typical output range.
    # For a real project, you would determine these from your test data.
    MODEL_MIN_SCORE = 10
    MODEL_MAX_SCORE = 95
    normalized_prediction = ((raw_prediction - MODEL_MIN_SCORE) / (MODEL_MAX_SCORE - MODEL_MIN_SCORE)) * 100
    
    # Clip the score to ensure it stays between 0 and 100
    return np.clip(normalized_prediction, 0, 100)


# ====================================================================
# Main App Interface
# ====================================================================
st.title("🧠 Personalized Learning Onboarding")

st.warning(
    "**Disclaimer:** This is a screening tool to help personalize your learning experience. "
    "It is **not a medical diagnosis**. Please consult a healthcare professional for an accurate assessment."
)

if model is None:
    st.error("**Error: Model files not found.** Please ensure all `.joblib` files are in the same directory.")
else:
    st.info("Please answer the questions below to help us tailor your learning journey.")

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
            # --- SCORING LOGIC ---
            symptom_score = calculate_symptom_score(answers)
            ai_score = get_ai_score(answers, model, scaler, feature_cols)
            
            # Blend the scores
            symptom_weight = 0.70  # 70% from the direct answers
            ai_weight = 0.30       # 30% from the AI model's analysis
            final_score = (symptom_score * symptom_weight) + (ai_score * ai_weight)

            st.write("---")
            st.subheader("Your Personalized Score")
            st.metric(label="Learning Attention Indicator", value=f"{final_score:.0f} / 100")

            if final_score > 70:
                st.error("High Indication: Your profile suggests a strong need for a highly structured and engaging learning environment.")
            elif final_score > 40:
                st.warning("Moderate Indication: Your profile suggests you may benefit from additional focus tools and personalized pacing.")
            else:
                st.success("Low Indication: Your profile suggests a more traditional learning environment may be suitable.")

    with st.expander("How is this score calculated?"):
        st.markdown("""
        Your final score is a **hybrid** of two methods to give a balanced view:
        - **70% Direct Score:** This part comes directly from your answers. It ensures that your score reflects what you've told us in an intuitive way.
        - **30% AI Analysis:** This part uses our machine learning model to look for deeper, underlying patterns that may not be obvious.
        
        This blended approach provides a more robust starting point for personalizing your education.
        """)