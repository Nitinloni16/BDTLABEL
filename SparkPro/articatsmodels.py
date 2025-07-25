import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
st.title("🤖 Model Comparison & Selection")

st.markdown("""
This dashboard streams and compares the results of multiple models to help you figure out the best model based on evaluation metrics.
""")

# Example: Replace this with your actual model results
# Each row is a model, columns are metrics (add/remove as needed)
model_results = [
    {"Model": "RandomForest", "AUC": 0.89, "RMSE": 0.32, "Accuracy": 0.85, "F1": 0.83},
    {"Model": "LogisticRegression", "AUC": 0.87, "RMSE": 0.35, "Accuracy": 0.82, "F1": 0.80},
    {"Model": "GradientBoosting", "AUC": 0.91, "RMSE": 0.30, "Accuracy": 0.87, "F1": 0.85},
    {"Model": "SVM", "AUC": 0.85, "RMSE": 0.38, "Accuracy": 0.80, "F1": 0.78},
    # Add more models and metrics as needed
]

df = pd.DataFrame(model_results)

st.subheader("📊 Model Evaluation Table")
st.dataframe(df, use_container_width=True)

# Highlight the best model(s) for each metric
def highlight_best(s, ascending=True):
    is_best = s == (s.min() if ascending else s.max())
    return ['background-color: #90ee90' if v else '' for v in is_best]

styled_df = df.style
if "AUC" in df.columns:
    styled_df = styled_df.apply(highlight_best, subset=["AUC"], ascending=False)
if "RMSE" in df.columns:
    styled_df = styled_df.apply(highlight_best, subset=["RMSE"], ascending=True)
if "Accuracy" in df.columns:
    styled_df = styled_df.apply(highlight_best, subset=["Accuracy"], ascending=False)
if "F1" in df.columns:
    styled_df = styled_df.apply(highlight_best, subset=["F1"], ascending=False)

st.subheader("🏆 Highlighted Best Models")
st.dataframe(styled_df, use_container_width=True)

st.success("✅ Model comparison table loaded successfully!")

