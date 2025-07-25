import streamlit as st
import pandas as pd
import os
import plotly.express as px
import re

st.set_page_config(page_title="Spark Job Results Visualization", layout="wide")
st.title("📊 Spark Job Results Visualization Dashboard")

# Path to outputs folder
output_dir = "E:/Project_LAB/Preprocess_DATASET/outputs"

# Only keep the approximate heat file for real-time heat analysis
approx_heat_path = "E:/Project_LAB/Preprocess_DATASET/outputs/approx_heat_windows.txt"

# Helper to load CSVs safely
def load_csv(filename):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.warning(f"File not found: {filename}")
        return None

# Helper to extract score from txt file
def extract_score(txt_file, metric_name):
    path = os.path.join(output_dir, txt_file)
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if metric_name in line:
                    found = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if found:
                        return float(found[-1])
    return None

# Helper to load only approximate heat metrics from txt file
def load_approx_heat_metrics():
    approx_data = []
    approx_raw = []
    if os.path.exists(approx_heat_path):
        with open(approx_heat_path, "r") as f:
            for line in f:
                approx_raw.append(line.rstrip())
                vals = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(vals) >= 2:
                    approx_data.append({
                        "Time": float(vals[0]),
                        "Approx_Heat": float(vals[1])
                    })
    approx_df = pd.DataFrame(approx_data) if approx_data else None
    return approx_df, approx_raw

# List of result files to visualize
result_files = [
    ("Overall Summary", "overall_summary.csv"),
    ("Top Machines", "top_machines.csv"),
    ("Scheduling Class Analysis", "scheduling_class_analysis.csv"),
    ("Collection Type Analysis", "collection_type_analysis.csv"),
    ("Priority Analysis", "priority_analysis.csv"),
    ("Task Failure Rate Prediction", "failure_prediction.csv"),
    ("Heat Produced Prediction", "heat_prediction.csv"),
    ("Energy Consumed Prediction", "energy_prediction.csv"),
    ("Task Execution Time Prediction", "duration_prediction.csv"),
    ("Realtime Heat Analysis Metrics", None),  # Only approximate_heat_windows.txt
]

# Mapping for metrics/score files
score_info = {
    "Task Failure Rate Prediction": ("auc_failure_score.txt", "AUC Score"),
    "Heat Produced Prediction": ("rmse_heat.txt", "RMSE"),
    "Energy Consumed Prediction": ("rmse_energy.txt", "RMSE"),
    "Task Execution Time Prediction": ("rmse_duration.txt", "RMSE"),
}

# Sidebar for selection
st.sidebar.header("Select Result to Visualize")
selected = st.sidebar.selectbox(
    "Choose a result table:",
    [name for name, _ in result_files]
)

# Load selected data
filename = dict(result_files).get(selected)
df = load_csv(filename) if filename else None

# Show score/metric if available
if selected in score_info:
    score_file, metric_name = score_info[selected]
    score = extract_score(score_file, metric_name)
    if score is not None:
        st.info(f"**{metric_name}**: `{score}`")
    else:
        st.warning(f"{metric_name} not found in {score_file}")

if selected == "Realtime Heat Analysis Metrics":
    st.subheader("Realtime Heat Analysis Metrics (from approx_heat_windows.txt)")
    approx_df, approx_raw = load_approx_heat_metrics()
    # Print the as-is results from the file
    st.markdown("**Raw Content of approx_heat_windows.txt:**")
    if approx_raw:
        st.code("\n".join(approx_raw), language="text")
    else:
        st.warning("No approx_heat_windows.txt data found.")
    # Show parsed table and plot
    if approx_df is not None:
        st.write("**Parsed Approximate Heat (Windows):**")
        st.dataframe(approx_df)
        fig = px.scatter(approx_df, x="Time", y="Approx_Heat", title="Approximate Heat Over Time (Scatter)")
        st.plotly_chart(fig, use_container_width=True)

elif df is not None:
    st.subheader(f"{selected}")
    st.dataframe(df)

    # Plotting logic for each result type using plotly.express (scatter plots where possible)
    if selected == "Overall Summary":
        df_plot = df.T.reset_index()
        df_plot.columns = ["Metric", "Value"]
        fig = px.scatter(df_plot, x="Metric", y="Value", title="Overall Summary Metrics (Scatter)")
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Top Machines":
        fig = px.scatter(
            df,
            x="Energy",
            y="Avg Heat",
            size="Tasks",
            color="Failure Rate (%)",
            hover_data=["machine_id"],
            title="Top Machines: Energy vs Avg Heat (Bubble size = Tasks, Color = Failure Rate)"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Scheduling Class Analysis":
        fig = px.scatter(df, x="scheduling_class", y="Heat", size="Tasks", color="CPU" if "CPU" in df.columns else "Heat",
                         title="Heat by Scheduling Class (Scatter)")
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Collection Type Analysis":
        fig = px.scatter(df, x="collection_type", y="Memory Usage", size="Tasks", color="Heat",
                         title="Memory Usage by Collection Type (Scatter)")
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Priority Analysis":
        fig = px.scatter(df, x="priority", y="Heat", size="Tasks" if "Tasks" in df.columns else None, color="Failure Rate (%)",
                         title="Heat by Priority (Scatter)")
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Task Failure Rate Prediction":
        if "failed" in df.columns and "prediction" in df.columns:
            cross = df.groupby(["failed", "prediction"]).size().reset_index(name="count")
            cross["failed"] = cross["failed"].astype(str)
            cross["prediction"] = cross["prediction"].astype(str)
            fig = px.bar(
                cross,
                x="failed",
                y="count",
                color="prediction",
                barmode="group",
                title="Actual vs Predicted Task Failure (Bar Plot)",
                labels={"failed": "Actual Failure", "prediction": "Predicted"}
            )
            st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(df, x="prediction", y="failed", title="Prediction vs Actual Task Failure (Scatter)")
        st.plotly_chart(fig2, use_container_width=True)

    elif selected == "Heat Produced Prediction":
        fig = px.scatter(df, x="heat", y="prediction", color="machine_id" if "machine_id" in df.columns else None,
                         title="Actual vs Predicted Heat Produced (Scatter)", labels={"heat": "Actual Heat", "prediction": "Predicted Heat"})
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Energy Consumed Prediction":
        fig = px.scatter(df, x="energy", y="prediction", color="machine_id" if "machine_id" in df.columns else None,
                         title="Actual vs Predicted Energy Consumed (Scatter)", labels={"energy": "Actual Energy", "prediction": "Predicted Energy"})
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Task Execution Time Prediction":
        fig = px.scatter(df, x="duration", y="prediction", color="machine_id" if "machine_id" in df.columns else None,
                         title="Actual vs Predicted Task Execution Time (Scatter)", labels={"duration": "Actual Duration", "prediction": "Predicted Duration"})
        st.plotly_chart(fig, use_container_width=True)

st.success("✅ Visualization code executed successfully!")