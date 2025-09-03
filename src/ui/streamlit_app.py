"""
Streamlit UI for the Autonomous ML Agent.

This module provides a web-based interface for interacting with
the autonomous ML system, including pipeline management and
model leaderboard visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Page configuration
st.set_page_config(
    page_title="Autonomous ML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000/api/v1"
REFRESH_INTERVAL = 30  # seconds

def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.title("🤖 Autonomous ML Agent")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Dashboard", "🚀 Create Pipeline", "📊 Model Leaderboard", "📈 Pipeline Status", "🔍 Model Explorer"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        if health_response.status_code == 200:
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.error("❌ API Error")
    except:
        st.sidebar.error("❌ API Unreachable")
    
    # Main content
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🚀 Create Pipeline":
        show_create_pipeline()
    elif page == "📊 Model Leaderboard":
        show_model_leaderboard()
    elif page == "📈 Pipeline Status":
        show_pipeline_status()
    elif page == "🔍 Model Explorer":
        show_model_explorer()

def show_dashboard():
    """Display the main dashboard."""
    st.title("🏠 Autonomous ML Agent Dashboard")
    st.markdown("Welcome to the Autonomous ML Agent - Your intelligent ML pipeline orchestrator!")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get system status
        status_response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            with col1:
                st.metric("Active Pipelines", status_data.get("agent", {}).get("active_pipelines", 0))
            
            with col2:
                st.metric("Completed Pipelines", status_data.get("agent", {}).get("completed_pipelines", 0))
            
            with col3:
                st.metric("Total Models", status_data.get("agent", {}).get("total_pipelines", 0))
            
            with col4:
                st.metric("System Status", "🟢 Running" if status_data.get("status") == "healthy" else "🔴 Error")
        
    except:
        # Fallback metrics
        with col1:
            st.metric("Active Pipelines", "N/A")
        with col2:
            st.metric("Completed Pipelines", "N/A")
        with col3:
            st.metric("Total Models", "N/A")
        with col4:
            st.metric("System Status", "❌ Unavailable")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Create New Pipeline", use_container_width=True):
            st.switch_page("🚀 Create Pipeline")
    
    with col2:
        if st.button("📊 View Leaderboard", use_container_width=True):
            st.switch_page("📊 Model Leaderboard")
    
    with col3:
        if st.button("📈 Monitor Pipelines", use_container_width=True):
            st.switch_page("📈 Pipeline Status")
    
    # Recent activity
    st.markdown("---")
    st.markdown("### Recent Activity")
    
    try:
        # Get recent pipelines
        pipelines_response = requests.get(f"{API_BASE_URL}/pipeline/", timeout=5)
        if pipelines_response.status_code == 200:
            pipelines = pipelines_response.json()
            
            if pipelines:
                # Create activity dataframe
                activity_data = []
                for pipeline in pipelines[:5]:  # Show last 5
                    activity_data.append({
                        "Pipeline ID": pipeline.get("pipeline_id", "N/A")[:8] + "...",
                        "Status": pipeline.get("status", "Unknown"),
                        "Created": pipeline.get("created_at", "N/A"),
                        "Type": pipeline.get("config", {}).get("task_type", "Unknown")
                    })
                
                if activity_data:
                    df = pd.DataFrame(activity_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent pipeline activity found.")
            else:
                st.info("No pipelines found. Create your first pipeline to get started!")
        
    except Exception as e:
        st.error(f"Failed to load recent activity: {str(e)}")
        st.info("No recent pipeline activity found.")

def show_create_pipeline():
    """Show the pipeline creation form."""
    st.title("🚀 Create New ML Pipeline")
    st.markdown("Upload your dataset and configure the autonomous ML pipeline.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=['csv', 'xlsx', 'parquet'],
        help="Supported formats: CSV, Excel, Parquet"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(file_details)
        
        with col2:
            # Preview data
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Data Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        
        # Pipeline configuration
        st.markdown("---")
        st.markdown("### Pipeline Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.text_input("Target Column", placeholder="Enter target column name")
            task_type = st.selectbox(
                "Task Type",
                ["classification", "regression", "multiclass", "binary"]
            )
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        with col2:
            optimization_metric = st.selectbox(
                "Optimization Metric",
                ["accuracy", "precision", "recall", "f1_score", "roc_auc", "mse", "mae", "r2_score"]
            )
            max_models = st.number_input("Maximum Models", value=10, min_value=1, max_value=20)
            enable_ensemble = st.checkbox("Enable Ensemble Methods", value=True)
            enable_meta_learning = st.checkbox("Enable Meta-Learning", value=True)
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                cross_validation_folds = st.number_input("CV Folds", value=5, min_value=3, max_value=10)
                max_runtime = st.number_input("Max Runtime (seconds)", value=3600, min_value=300, max_value=7200)
            
            with col2:
                feature_selection = st.checkbox("Enable Feature Selection", value=True)
                data_cleaning = st.checkbox("Enable Data Cleaning", value=True)
                feature_engineering = st.checkbox("Enable Feature Engineering", value=True)
        
        # Create pipeline button
        if st.button("🚀 Create Pipeline", type="primary", use_container_width=True):
            if target_column:
                create_pipeline(
                    uploaded_file, target_column, task_type, optimization_metric,
                    test_size, random_state, cross_validation_folds, max_models,
                    enable_ensemble, enable_meta_learning, max_runtime,
                    feature_selection, data_cleaning, feature_engineering
                )
            else:
                st.error("Please specify a target column.")

def create_pipeline(file, target_column, task_type, optimization_metric, **kwargs):
    """Create a new ML pipeline."""
    st.info("🚀 Creating pipeline... This may take a few moments.")
    
    try:
        # Prepare form data
        files = {"dataset": file}
        data = {
            "target_column": target_column,
            "task_type": task_type,
            "optimization_metric": optimization_metric,
            "test_size": kwargs.get("test_size", 0.2),
            "random_state": kwargs.get("random_state", 42),
            "cross_validation_folds": kwargs.get("cross_validation_folds", 5),
            "max_models": kwargs.get("max_models", 10),
            "enable_ensemble": kwargs.get("enable_ensemble", True),
            "max_runtime": kwargs.get("max_runtime", 3600),
            "enable_meta_learning": kwargs.get("enable_meta_learning", True)
        }
        
        # Create pipeline
        response = requests.post(
            f"{API_BASE_URL}/pipeline/create",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            pipeline_id = result.get("pipeline_id")
            
            st.success(f"✅ Pipeline created successfully!")
            st.info(f"Pipeline ID: {pipeline_id}")
            
            # Show next steps
            st.markdown("---")
            st.markdown("### Next Steps")
            st.markdown("1. **Monitor Progress**: Check the Pipeline Status page")
            st.markdown("2. **View Results**: Once complete, see the Model Leaderboard")
            st.markdown("3. **Export Model**: Download the trained pipeline")
            
            # Auto-redirect option
            if st.button("📈 Go to Pipeline Status"):
                st.switch_page("📈 Pipeline Status")
        
        else:
            st.error(f"❌ Failed to create pipeline: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error creating pipeline: {str(e)}")

def show_model_leaderboard():
    """Display the model leaderboard."""
    st.title("📊 Model Leaderboard")
    st.markdown("Compare model performance and find the best models for your task.")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pipeline_filter = st.selectbox("Filter by Pipeline", ["All Pipelines", "Recent", "Completed"])
    
    with col2:
        sort_by = st.selectbox("Sort by", ["accuracy", "precision", "recall", "f1_score", "training_time"])
    
    with col3:
        limit = st.number_input("Show Top N Models", value=20, min_value=5, max_value=100)
    
    # Load leaderboard data
    try:
        response = requests.get(
            f"{API_BASE_URL}/models/leaderboard",
            params={"limit": limit, "sort_by": sort_by},
            timeout=10
        )
        
        if response.status_code == 200:
            leaderboard_data = response.json()
            
            if leaderboard_data.get("models"):
                models = leaderboard_data["models"]
                
                # Create performance dataframe
                performance_data = []
                for model in models:
                    performance_data.append({
                        "Model": model.get("model_name", "Unknown"),
                        "Type": model.get("model_type", "Unknown"),
                        "Accuracy": model.get("performance_metrics", {}).get("accuracy", 0),
                        "Precision": model.get("performance_metrics", {}).get("precision", 0),
                        "Recall": model.get("performance_metrics", {}).get("recall", 0),
                        "F1 Score": model.get("performance_metrics", {}).get("f1_score", 0),
                        "Training Time": f"{model.get('training_time', 0):.2f}s"
                    })
                
                df = pd.DataFrame(performance_data)
                
                # Display leaderboard
                st.dataframe(df, use_container_width=True)
                
                # Performance visualization
                st.markdown("---")
                st.markdown("### Performance Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy comparison
                    fig_acc = px.bar(
                        df, x="Model", y="Accuracy",
                        title="Model Accuracy Comparison",
                        color="Type"
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    # Training time comparison
                    fig_time = px.bar(
                        df, x="Model", y="Training Time",
                        title="Training Time Comparison",
                        color="Type"
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Best model highlight
                if leaderboard_data.get("best_model"):
                    best_model = leaderboard_data["best_model"]
                    st.markdown("---")
                    st.markdown("### 🏆 Best Performing Model")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", best_model.get("model_name", "Unknown"))
                    with col2:
                        st.metric("Type", best_model.get("model_type", "Unknown"))
                    with col3:
                        st.metric("Accuracy", f"{best_model.get('performance_metrics', {}).get('accuracy', 0):.4f}")
            
            else:
                st.info("No models found in the leaderboard.")
        
        else:
            st.error(f"Failed to load leaderboard: {response.text}")
    
    except Exception as e:
        st.error(f"Error loading leaderboard: {str(e)}")
        st.info("No models found in the leaderboard.")

def show_pipeline_status():
    """Display pipeline status and monitoring."""
    st.title("📈 Pipeline Status")
    st.markdown("Monitor the status and progress of your ML pipelines.")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=True)
    
    if auto_refresh:
        time.sleep(0.1)  # Small delay for refresh
    
    # Load pipeline data
    try:
        response = requests.get(f"{API_BASE_URL}/pipeline/", timeout=10)
        
        if response.status_code == 200:
            pipelines = response.json()
            
            if pipelines:
                # Group pipelines by status
                active_pipelines = [p for p in pipelines if p.get("status") == "active"]
                completed_pipelines = [p for p in pipelines if p.get("status") == "completed"]
                failed_pipelines = [p for p in pipelines if p.get("status") == "failed"]
                
                # Status overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Pipelines", len(pipelines))
                with col2:
                    st.metric("Active", len(active_pipelines), delta=len(active_pipelines))
                with col3:
                    st.metric("Completed", len(completed_pipelines), delta=len(completed_pipelines))
                with col4:
                    st.metric("Failed", len(failed_pipelines), delta=-len(failed_pipelines))
                
                # Active pipelines
                if active_pipelines:
                    st.markdown("---")
                    st.markdown("### 🔄 Active Pipelines")
                    
                    for pipeline in active_pipelines:
                        with st.expander(f"Pipeline {pipeline.get('pipeline_id', 'N/A')[:8]}..."):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Status:** {pipeline.get('status', 'Unknown')}")
                                st.write(f"**Target:** {pipeline.get('target_column', 'N/A')}")
                                st.write(f"**Task:** {pipeline.get('config', {}).get('task_type', 'Unknown')}")
                            
                            with col2:
                                st.write(f"**Created:** {pipeline.get('created_at', 'N/A')}")
                                st.write(f"**Progress:** {pipeline.get('progress', 'N/A')}")
                                
                                # Progress bar
                                progress = pipeline.get("progress", 0)
                                if isinstance(progress, (int, float)):
                                    st.progress(progress / 100)
                
                # Completed pipelines
                if completed_pipelines:
                    st.markdown("---")
                    st.markdown("### ✅ Completed Pipelines")
                    
                    completed_data = []
                    for pipeline in completed_pipelines:
                        completed_data.append({
                            "Pipeline ID": pipeline.get("pipeline_id", "N/A")[:8] + "...",
                            "Target": pipeline.get("target_column", "N/A"),
                            "Task": pipeline.get("config", {}).get("task_type", "Unknown"),
                            "Completed": pipeline.get("completed_at", "N/A"),
                            "Status": pipeline.get("status", "Unknown")
                        })
                    
                    if completed_data:
                        df = pd.DataFrame(completed_data)
                        st.dataframe(df, use_container_width=True)
                
                # Failed pipelines
                if failed_pipelines:
                    st.markdown("---")
                    st.markdown("### ❌ Failed Pipelines")
                    
                    for pipeline in failed_pipelines:
                        with st.expander(f"Pipeline {pipeline.get('pipeline_id', 'N/A')[:8]}..."):
                            st.error(f"**Error:** {pipeline.get('error', 'Unknown error')}")
                            st.write(f"**Failed at:** {pipeline.get('failed_at', 'N/A')}")
                            
                            if st.button(f"Retry Pipeline {pipeline.get('pipeline_id', 'N/A')[:8]}..."):
                                st.info("Retry functionality not yet implemented.")
            
            else:
                st.info("No pipelines found. Create your first pipeline to get started!")
        
        else:
            st.error(f"Failed to load pipeline status: {response.text}")
    
    except Exception as e:
        st.error(f"Error loading pipeline status: {str(e)}")
        st.info("No pipelines found.")

def show_model_explorer():
    """Show detailed model exploration and analysis."""
    st.title("🔍 Model Explorer")
    st.markdown("Explore detailed information about trained models and their performance.")
    
    st.info("🔧 Model Explorer functionality is under development.")
    st.markdown("This section will include:")
    st.markdown("- Detailed model performance analysis")
    st.markdown("- Feature importance visualization")
    st.markdown("- Model comparison tools")
    st.markdown("- Hyperparameter analysis")
    st.markdown("- Model interpretability tools")

if __name__ == "__main__":
    main()
