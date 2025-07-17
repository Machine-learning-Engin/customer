import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import requests  # For OpenRouter API
import json

# Title
st.title("📉 Telco Customer Churn Predictor with LIME + LLM Recommendations")

# Load model
model = joblib.load("telco_churn_model.pkl")

# OpenRouter API Config
OPENROUTER_API_KEY = "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"
LLM_MODEL = "qwen/qwen3-32b:free"

# ➕ Format LIME explanation as text
def lime_explanation_to_text(explanation):
    try:
        weights = explanation.as_list()
        explanation_text = "\n".join([f"- {feature}: {round(weight, 4)}" for feature, weight in weights])
        return explanation_text
    except Exception as e:
        return f"Error extracting LIME explanation: {e}"

# 🔁 Updated LLM function to accept lime explanation
def get_llm_recommendation(customer_features, lime_reasoning=None):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "Telco Churn App"
    }

    # Compose full prompt
    prompt = f"""
A customer is predicted to churn. Based on the following features and explanation of why the churn is likely, provide personalized and actionable recommendations to retain this customer:

📌 Customer Features:
{json.dumps(customer_features, indent=2)}

📊 LIME Explanation of Churn Risk:
{lime_reasoning or "N/A"}
"""

    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a customer retention expert."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error fetching recommendation: {e}"

# Upload file
uploaded_file = st.file_uploader("Upload your customer data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    original_df = df.copy()

    def preprocess(df):
        df = df.drop(columns=['customerID'], errors='ignore')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        categorical = df.select_dtypes(include='object').columns
        df[categorical] = df[categorical].astype('category')
        df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
        return df

    df_processed = preprocess(df)
    predictions = model.predict(df_processed)
    churn_prob = model.predict_proba(df_processed)[:, 1]

    result_df = original_df.loc[df_processed.index].copy()
    result_df['Churn Prediction'] = predictions
    result_df['Churn Probability'] = churn_prob

    churned = result_df[result_df['Churn Prediction'] == 1]

    st.subheader("📊 Customers Predicted to Churn")
    st.write(churned)

    # LIME Explainer Setup
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(df_processed),
        feature_names=df_processed.columns.tolist(),
        class_names=['No Churn', 'Churn'],
        mode='classification'
    )

    st.subheader("🧠 LIME Explanation for a Churned Customer")
    churned_index = st.number_input("Pick a churned customer index to explain", min_value=0, max_value=len(churned)-1, step=1)

    churned_row = churned.iloc[churned_index]
    original_index = churned_row.name
    input_data = df_processed.loc[original_index].values

    explanation = explainer.explain_instance(
        input_data,
        model.predict_proba,
        num_features=10
    )

    # ➕ Convert explanation to text
    explanation_text = lime_explanation_to_text(explanation)

    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)
    st.markdown("✅ Top features influencing churn shown above.")

    # LLM Retention Recommendation
    st.subheader("🛠️ AI Recommendation to Retain This Customer")
    with st.spinner("Fetching AI-generated retention strategy..."):
        selected_features = dict(zip(df_processed.columns, input_data))
        advice = get_llm_recommendation(selected_features, lime_reasoning=explanation_text)
        st.success("Recommendation received:")
        st.markdown(advice)
