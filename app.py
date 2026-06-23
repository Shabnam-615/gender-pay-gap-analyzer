import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Gender Pay Gap Analyzer", layout="wide")
st.title("Gender Pay Gap Analyzer")
st.caption("ML-powered salary fairness analysis with SHAP explainability")

model = joblib.load('model.pkl')
df = pd.read_csv('data/Glassdoor Gender Pay Gap.csv')

le = LabelEncoder()
df['JobTitle_enc'] = le.fit_transform(df['JobTitle'])
df['Dept_enc'] = le.fit_transform(df['Dept'])
df['Education_enc'] = le.fit_transform(df['Education'])

# --- Tab layout ---
tab1, tab2 = st.tabs(["Pay gap overview", "What-if simulator"])

with tab1:
    st.subheader("Average salary by gender")
    avg = df.groupby('Gender')['BasePay'].mean()
    st.bar_chart(avg)
    gap = avg['Male'] - avg['Female']
    st.metric("Raw pay gap", f"${gap:,.0f}", delta="-8,514 per year for women", delta_color="inverse")

    st.subheader("Pay gap by department")
    dept_gap = df.groupby(['Dept','Gender'])['BasePay'].mean().unstack()
    dept_gap['Gap'] = dept_gap['Male'] - dept_gap['Female']
    st.bar_chart(dept_gap['Gap'])

with tab2:
    st.subheader("Predict salary & see why")
    col1, col2 = st.columns(2)

    with col1:
        job = st.selectbox("Job title", sorted(df['JobTitle'].unique()))
        dept = st.selectbox("Department", sorted(df['Dept'].unique()))
        edu = st.selectbox("Education", sorted(df['Education'].unique()))

    with col2:
        age = st.slider("Age", 18, 65, 35)
        seniority = st.slider("Seniority", 1, 10, 5)
        perf = st.slider("Performance rating", 1, 5, 3)

    if st.button("Predict salary"):
        job_enc = sorted(df['JobTitle'].unique()).index(job)
        dept_enc = sorted(df['Dept'].unique()).index(dept)
        edu_enc = sorted(df['Education'].unique()).index(edu)

        input_df = pd.DataFrame([[job_enc, age, perf, edu_enc, dept_enc, seniority]],
                                 columns=['JobTitle_enc','Age','PerfEval','Education_enc','Dept_enc','Seniority'])

        pred = model.predict(input_df)[0]
        st.metric("Predicted salary", f"${pred:,.0f}")

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(input_df)

        st.subheader("Why this salary? (SHAP breakdown)")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_vals[0], show=False)
        st.pyplot(fig)