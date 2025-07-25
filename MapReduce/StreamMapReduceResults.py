import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Cloud Workload Heat Analysis Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
    }
    .stMetric > label {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

def load_results():
    """Load results from JSON file"""
    try:
        with open('mapreduce_results.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ Results file not found. Please run the MapReduce analysis first!")
        return None
    except json.JSONDecodeError:
        st.error("❌ Error reading results file. Please check the file format.")
        return None

def create_machine_dataframe(results):
    """Create DataFrame for machine analysis"""
    machine_data = []
    for key, stats in results.items():
        if key.startswith('machine_'):
            machine_data.append({
                'Machine ID': key.replace('machine_', ''),
                'Tasks': stats['count'],
                'Avg CPU (%)': stats['avg_cpu_usage'],
                'Peak CPU (%)': stats['avg_max_cpu'],
                'Memory (MB)': stats['avg_memory'],
                'Memory Usage (%)': stats['avg_memory_usage'],
                'Heat (°C)': stats['avg_heat_generation'],
                'Energy (kWh)': stats['total_energy_consumption'],
                'Duration (sec)': stats['avg_duration'],
                'Failure Rate (%)': stats['failure_rate'],
                'Failed Tasks': stats['failed_tasks']
            })
    return pd.DataFrame(machine_data)

def create_class_dataframe(results):
    """Create DataFrame for scheduling class analysis"""
    class_data = []
    for key, stats in results.items():
        if key.startswith('class_'):
            class_data.append({
                'Scheduling Class': key.replace('class_', ''),
                'Tasks': stats['count'],
                'Avg CPU (%)': stats['avg_cpu_usage'],
                'Heat (°C)': stats['avg_heat_generation'],
                'Energy (kWh)': stats['total_energy_consumption'],
                'Failure Rate (%)': stats['failure_rate']
            })
    return pd.DataFrame(class_data)

def create_priority_dataframe(results):
    """Create DataFrame for priority analysis"""
    priority_data = []
    for key, stats in results.items():
        if key.startswith('priority_'):
            priority_data.append({
                'Priority': key.replace('priority_', ''),
                'Tasks': stats['count'],
                'Heat (°C)': stats['avg_heat_generation'],
                'Energy (kWh)': stats['total_energy_consumption'],
                'Failure Rate (%)': stats['failure_rate']
            })
    return pd.DataFrame(priority_data)

def create_collection_type_dataframe(results):
    """Create DataFrame for collection type analysis"""
    collection_data = []
    for key, stats in results.items():
        if key.startswith('collection_type_'):
            collection_data.append({
                'Collection Type': key.replace('collection_type_', ''),
                'Tasks': stats['count'],
                'Heat (°C)': stats['avg_heat_generation'],
                'Memory Usage (%)': stats['avg_memory_usage'],
                'Energy (kWh)': stats['total_energy_consumption']
            })
    return pd.DataFrame(collection_data)

def main():
    st.title("🔥 Cloud Workload Heat Analysis Dashboard")
    st.markdown("Real-time visualization of cloud workload performance metrics")
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Load data
    data = load_results()
    if data is None:
        st.stop()
    
    results = data['analysis_results']
    timestamp = data.get('timestamp', 'Unknown')
    row_count = data.get('row_count', 0)
    
    # Header metrics
    st.markdown("### 📈 Overall Summary")
    if 'overall' in results:
        overall = results['overall']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Tasks", f"{overall['count']:,}")
        with col2:
            st.metric("Avg CPU Usage", f"{overall['avg_cpu_usage']:.2f}%")
        with col3:
            st.metric("Heat Generation", f"{overall['avg_heat_generation']:.1f}°C")
        with col4:
            st.metric("Energy Consumption", f"{overall['total_energy_consumption']:.2f} kWh")
        with col5:
            st.metric("Failure Rate", f"{overall['failure_rate']:.1f}%")
    
    # Analysis type selector
    analysis_type = st.sidebar.selectbox(
        "🔍 Select Analysis Type",
        ["Machine Analysis", "Scheduling Class", "Priority Analysis", "Collection Type"]
    )
    
    # Machine Analysis
    if analysis_type == "Machine Analysis":
        st.markdown("### 🖥️ Machine-wise Performance Analysis")
        
        machine_df = create_machine_dataframe(results)
        if not machine_df.empty:
            # Top machines by heat generation
            top_n = st.sidebar.slider("Top N Machines", 5, min(20, len(machine_df)), 10)
            top_machines = machine_df.nlargest(top_n, 'Heat (°C)')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Heat generation chart
                fig_heat = px.bar(
                    top_machines, 
                    x='Machine ID', 
                    y='Heat (°C)',
                    title=f"Top {top_n} Machines by Heat Generation",
                    color='Heat (°C)',
                    color_continuous_scale='Reds'
                )
                fig_heat.update_layout(height=400)
                st.plotly_chart(fig_heat, use_container_width=True)
            
            with col2:
                # Energy consumption vs Tasks
                fig_energy = px.scatter(
                    top_machines,
                    x='Tasks',
                    y='Energy (kWh)',
                    size='Heat (°C)',
                    color='Failure Rate (%)',
                    hover_data=['Machine ID'],
                    title="Energy Consumption vs Task Count"
                )
                fig_energy.update_layout(height=400)
                st.plotly_chart(fig_energy, use_container_width=True)
            
            # CPU and Memory utilization
            col3, col4 = st.columns(2)
            
            with col3:
                # CPU utilization
                fig_cpu = px.box(
                    machine_df.head(20),
                    y='Avg CPU (%)',
                    title="CPU Utilization Distribution"
                )
                fig_cpu.update_layout(height=300)
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col4:
                # Memory usage
                fig_mem = px.histogram(
                    machine_df,
                    x='Memory Usage (%)',
                    nbins=20,
                    title="Memory Usage Distribution"
                )
                fig_mem.update_layout(height=300)
                st.plotly_chart(fig_mem, use_container_width=True)
            
            # Detailed table
            st.markdown("### 📋 Detailed Machine Statistics")
            st.dataframe(machine_df, use_container_width=True)
    
    # Scheduling Class Analysis
    elif analysis_type == "Scheduling Class":
        st.markdown("### 📋 Scheduling Class Performance")
        
        class_df = create_class_dataframe(results)
        if not class_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Tasks by scheduling class
                fig_tasks = px.pie(
                    class_df,
                    values='Tasks',
                    names='Scheduling Class',
                    title="Task Distribution by Scheduling Class"
                )
                st.plotly_chart(fig_tasks, use_container_width=True)
            
            with col2:
                # Heat vs Failure rate
                fig_heat_fail = px.scatter(
                    class_df,
                    x='Heat (°C)',
                    y='Failure Rate (%)',
                    size='Tasks',
                    color='Scheduling Class',
                    title="Heat Generation vs Failure Rate"
                )
                st.plotly_chart(fig_heat_fail, use_container_width=True)
            
            # Performance comparison
            fig_compare = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage (%)', 'Heat Generation (°C)', 
                               'Energy Consumption (kWh)', 'Failure Rate (%)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_compare.add_trace(
                go.Bar(x=class_df['Scheduling Class'], y=class_df['Avg CPU (%)'], name='CPU'),
                row=1, col=1
            )
            fig_compare.add_trace(
                go.Bar(x=class_df['Scheduling Class'], y=class_df['Heat (°C)'], name='Heat'),
                row=1, col=2
            )
            fig_compare.add_trace(
                go.Bar(x=class_df['Scheduling Class'], y=class_df['Energy (kWh)'], name='Energy'),
                row=2, col=1
            )
            fig_compare.add_trace(
                go.Bar(x=class_df['Scheduling Class'], y=class_df['Failure Rate (%)'], name='Failure'),
                row=2, col=2
            )
            
            fig_compare.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_compare, use_container_width=True)
            
            st.dataframe(class_df, use_container_width=True)
    
    # Priority Analysis
    elif analysis_type == "Priority Analysis":
        st.markdown("### 🎯 Priority-based Performance Analysis")
        
        priority_df = create_priority_dataframe(results)
        if not priority_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Priority vs Heat
                fig_priority_heat = px.bar(
                    priority_df,
                    x='Priority',
                    y='Heat (°C)',
                    color='Tasks',
                    title="Heat Generation by Priority Level"
                )
                st.plotly_chart(fig_priority_heat, use_container_width=True)
            
            with col2:
    # Priority vs Failure Rate (changed from line to scatter)
                fig_priority_fail = px.scatter(
                    priority_df,
                    x='Priority',
                    y='Failure Rate (%)',
                    size='Tasks',
                    color='Failure Rate (%)',
                    hover_name='Priority',
                    title="Failure Rate by Priority Level (Scatter Plot)",
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_priority_fail, use_container_width=True)

            st.dataframe(priority_df, use_container_width=True)
    
    # Collection Type Analysis
    elif analysis_type == "Collection Type":
        st.markdown("### 📦 Collection Type Performance")
        
        collection_df = create_collection_type_dataframe(results)
        if not collection_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Collection type distribution
                fig_collection = px.treemap(
                    collection_df,
                    path=['Collection Type'],
                    values='Tasks',
                    color='Heat (°C)',
                    title="Collection Types by Task Count"
                )
                st.plotly_chart(fig_collection, use_container_width=True)
            
            with col2:
                # Memory usage by collection type
                fig_mem_usage = px.bar(
                    collection_df,
                    x='Collection Type',
                    y='Memory Usage (%)',
                    color='Energy (kWh)',
                    title="Memory Usage by Collection Type"
                )
                st.plotly_chart(fig_mem_usage, use_container_width=True)
            
            st.dataframe(collection_df, use_container_width=True)
    
    # Footer information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📅 Last Updated: {timestamp}")
    with col2:
        st.info(f"📊 Rows Processed: {row_count:,}")
    with col3:
        st.info(f"🔄 Analysis Categories: {len([k for k in results.keys() if not k.startswith('overall')])}")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

    