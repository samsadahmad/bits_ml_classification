"""
BITS ML Classification Models Streamlit Application
Interactive interface to test multiple classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
import base64
import subprocess
import sys

TABLE_SCROLL_THRESHOLD_ROWS = 10+2  # 2 extra rows for header and padding

# Set page config
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load and encode logo as base64
logo_path = Path("assets/bits-pilani-logo.png")
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    logo_url = f"data:image/png;base64,{logo_base64}"
else:
    logo_url = ""

# Selectors are duplicated across Streamlit DOM variants because tab markup differs by version.
st.markdown(
    f"""
    <style>
    .logo-header {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: white;
        border-bottom: 1px solid #dfe4ff;
        padding: 0.75rem 2rem;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}
    .logo-header img {{
        height: 60px;
        margin-right: 1rem;
    }}
    .header-title {{
        flex-grow: 1;
        margin: 0;
    }}
    .header-title h1 {{
        margin: 0;
        font-size: 1.5rem;
        color: #5e49e2;
    }}
    /* Add top margin to main content to account for fixed header */
    .main {{
        margin-top: 20px !important;
    }}
    .dashboard-header {{
        position: sticky;
        top: 20px;
        z-index: 30;
        left: 50%;
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(4px);
        padding: 0.5rem 1rem 0.7rem;
        overflow: hidden;
        isolation: isolate;
    }}
    .dashboard-header::before {{
        content: '';
        position: absolute;
        inset: 0;
        background-image: url('{logo_url}');
        background-repeat: no-repeat;
        background-position: center;
        background-size: min(100px, 55vw);
        opacity: 0.10;
        pointer-events: none;
        z-index: 0;
    }}
    .dashboard-header > * {{
        position: relative;
        z-index: 1;
    }}
    /* Tab list styling */
    [role="tablist"] {{
        gap: 8px !important;
        border-bottom: 2px solid #dfe4ff !important;
        background-color: transparent !important;
        padding-bottom: 8px !important;
        margin-bottom: 0 !important;
        position: sticky;
        top: 20px;
        z-index: 20;
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(4px);
    }}
    /* Tab highlight/border */
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"] {{
        background-color: #5e49e2 !important;
        background-image: none !important;
        bottom: 0 !important;
    }}
    /* Tab buttons */
    [role="tab"] {{
        background-color: #ffffff !important;
        border: 2px solid #5e49e2 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 20px !important;
        margin: 0 4px !important;
        min-height: 44px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.3s ease !important;
        color: #5e49e2 !important;
        cursor: pointer;
    }}
    [role="tab"] p,
    [role="tab"] span {{
        margin: 0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        color: #5e49e2 !important;
    }}
    /* Tab hover state */
    [role="tab"]:hover {{
        background-color: #eef2ff !important;
        border-color: #5e49e2 !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(94, 73, 226, 0.15) !important;
    }}
    /* Active tab state */
    [role="tab"][aria-selected="true"] {{
        background-color: #5e49e2 !important;
        border-color: #5e49e2 !important;
        box-shadow: 0 4px 16px rgba(94, 73, 226, 0.3) !important;
        transform: translateY(-2px);
    }}
    [role="tab"][aria-selected="true"],
    [role="tab"][aria-selected="true"] *,
    [role="tab"][aria-selected="true"] p,
    [role="tab"][aria-selected="true"] span {{
        color: #ffffff !important;
    }}
    /* Tab panel full width scrolling */
    [role="tabpanel"] {{
        width: 100% !important;
        max-width: 100% !important;
        max-height: none !important;
        height: auto !important;
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    /* Remove unnecessary block container padding */
    .block-container {{
        padding-top: 0 !important;
        padding-bottom: 5rem !important;
        padding-left: 6rem !important;
        padding-right: 6rem !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}
    /* Main content area */
    .main {{
        margin-top: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }}
    /* Center stApp */
    .stApp {{
        display: flex !important;
        justify-content: center !important;
    }}
    /* Override default red selection indicator to blue */
    .st-emotion-cache-qksclw[data-selected] .react-aria-SelectionIndicator {{
        background-color: #2563eb !important;
    }}
    [data-selected] .react-aria-SelectionIndicator {{
        background-color: #2563eb !important;
    }}
    /* Override red selectbox tag to blue */
    .st-emotion-cache-14c3ugh {{
        background-color: #2563eb !important;
        color: rgb(255, 255, 255) !important;
    }}
    /* Ensure footer is visible */
    footer {{
        visibility: visible !important;
        display: block !important;
    }}
    [data-testid="stFooter"] {{
        visibility: visible !important;
        display: block !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom styling
if logo_url:
    st.markdown("""
        <style>
        .main {{
            padding: 0 !important;
            margin: 0 !important;
            position: relative;
            width: 100% !important;
        }}
        .main::before {{
            content: '';
            position: fixed;
            top: 0%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 300px;
            height: 300px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
        }}
        h1 {{
            color: #5e49e2;
            margin-top: 0rem;
            margin-bottom: 0.25rem;
            padding: 0.25rem 0;
        }}
        h2 {{
            color: #ff7f0e;
        }}
        hr {{
            margin: 0.25rem 0;
            padding: 0;
        }}
        /* Dataframe header styling */
        [data-testid="stDataFrame"] thead th {{
            background-color: #5e49e2 !important;
            color: white !important;
            font-weight: bold !important;
        }}
        div[data-testid="stDataFrame"] thead tr {{
            background-color: #5e49e2 !important;
        }}
        /* Dataframe body styling */
        [data-testid="stDataFrame"] tbody tr {{
            background-color: #f6f8ff !important;
        }}
        [data-testid="stDataFrame"] tbody tr:nth-child(even) {{
            background-color: #edf1ff !important;
        }}
        [data-testid="stDataFrame"] tbody td {{
            color: #1f2937 !important;
        }}
        /* Override red selection indicator to blue */
        .st-emotion-cache-qksclw[data-selected] .react-aria-SelectionIndicator {{
            background-color: #2563eb !important;
        }}
        [data-selected] .react-aria-SelectionIndicator {{
            background-color: #2563eb !important;
        }}
        /* Override red selectbox tag to blue */
        .st-emotion-cache-14c3ugh {{
            background-color: #2563eb !important;
            color: rgb(255, 255, 255) !important;
        }}
        /* Selectbox button styling */
        [data-testid="stSelectbox"] button {{
            background-color: #2563eb !important;
            color: white !important;
        }}
        /* Ensure footer is visible */
        footer {{
            visibility: visible !important;
            display: block !important;
        }}
        [data-testid="stFooter"] {{
            visibility: visible !important;
            display: block !important;
        }}
        </style>
        """, unsafe_allow_html=True)


def auto_train_models_if_missing():
    """
    Automatically train models if they don't exist.
    This ensures the app works on Streamlit Cloud on first run.
    """
    model_dir = Path("model")
    required_files = [
        "logistic_regression.pkl",
        "decision_tree.pkl", 
        "k-nearest_neighbor.pkl",
        "naive_bayes.pkl",
        "random_forest.pkl",
        "scaler.pkl"
    ]
    
    # Check if all model files exist
    all_exist = all((model_dir / file).exists() for file in required_files)
    
    if not all_exist:
        st.warning("⏳ Models not found. Training models for the first time... This may take 30-60 seconds.")
        progress_bar = st.progress(0)
        
        try:
            # Import and run training pipeline
            from train_models import MLClassificationPipeline
            
            pipeline = MLClassificationPipeline()
            
            progress_bar.progress(10)
            st.info("📊 Loading dataset...")
            X, y = pipeline.load_dataset()
            
            progress_bar.progress(30)
            st.info("🔧 Preprocessing data...")
            pipeline.preprocess_data(X, y)
            
            progress_bar.progress(50)
            st.info("🤖 Training models...")
            pipeline.train_all_models()
            
            progress_bar.progress(80)
            st.info("💾 Saving models and results...")
            pipeline.save_models()
            pipeline.save_results_csv()
            pipeline.save_test_data()
            
            progress_bar.progress(100)
            st.success("✅ Models trained successfully! Reloading app...")
            
            # Rerun to load the newly trained models
            import time
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during model training: {str(e)}")
            st.info("Please try refreshing the page or running: `python train_models.py` locally")


@st.cache_resource
def load_models():
    """Load all trained models from pickle files"""
    models = {}
    model_dir = Path("model")
    
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbor": "k-nearest_neighbor.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl"
    }
    
    for model_name, filename in model_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    return models


@st.cache_resource
def load_scaler():
    """Load the StandardScaler used for training"""
    scaler_path = Path("model/scaler.pkl")
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_results():
    """Load model results from CSV"""
    results_path = Path("model_results.csv")
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


@st.cache_data
def load_test_data():
    """Load test data"""
    test_path = Path("test_data.csv")
    if test_path.exists():
        return pd.read_csv(test_path)
    return None


def display_metrics_table(results_df):
    """Display metrics comparison table"""
    st.subheader("📊 Model Performance Comparison")
    
    # Round metrics for display
    display_df = results_df.copy()
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    table_html = display_df.to_html(index=False, classes="metrics-table")
    st.markdown(
        """
        <style>
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        .metrics-table thead th {
            background-color: #5e49e2;
            color: #ffffff;
            font-weight: 700;
            padding: 10px;
            border: 1px solid #d9dcee;
            text-align: left;
        }
        .metrics-table tbody td {
            color: #1f2937;
            padding: 8px 10px;
            border: 1px solid #e4e7f2;
        }
        .metrics-table tbody tr:nth-child(odd) {
            background-color: #f6f8ff;
        }
        .metrics-table tbody tr:nth-child(even) {
            background-color: #edf1ff;
        }
        [data-testid="stMultiSelect"] [data-baseweb="select"] {
            background-color: #2768F5;
            border-radius: 10px;
        }
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background-color: #f6f8ff;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background-color: #228B22;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(
            results_df,
            x='Model',
            y='Accuracy',
            title='Accuracy Comparison',
            labels={'Accuracy': 'Accuracy Score'},
            color='Accuracy',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F1 Score comparison
        fig = px.bar(
            results_df,
            x='Model',
            y='F1',
            title='F1 Score Comparison',
            labels={'F1': 'F1 Score'},
            color='F1',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # All metrics radar chart
    st.subheader("Radar Chart - All Metrics Comparison")
    
    models_list = results_df['Model'].tolist()
    metrics_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    # Select models to compare
    selected_models = st.multiselect(
        "Select models to compare:",
        models_list,
        default=models_list
    )
    
    if selected_models:
        filtered_df = results_df[results_df['Model'].isin(selected_models)]
        
        fig = go.Figure()
        
        for _, row in filtered_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[col] for col in metrics_cols],
                theta=metrics_cols,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def display_dataset_upload():
    """Display dataset upload functionality"""
    st.subheader("📤 Upload Test Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file (test data)",
        type="csv",
        help="Upload only test data. Format: same features as training data"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write(f"Dataset shape: {df.shape}")
        st.dataframe(df.head(), use_container_width=True)
        return df
    
    return None


def display_scrollable_table(df, max_rows=TABLE_SCROLL_THRESHOLD_ROWS, row_height=35, show_serial_number=True):
    """Display a dataframe with scrolling enabled after max_rows records."""
    visible_rows = min(len(df), max_rows)
    enable_vertical_scroll = len(df) > max_rows
    table_height = (visible_rows + 1) * row_height if enable_vertical_scroll else None

    # st.dataframe renders to a canvas grid, so header CSS only works on an HTML table.
    st.markdown(
        """
        <style>
        .scroll-table-wrapper {
            overflow-x: auto;
            border: 1px solid #d9dcee;
            border-radius: 6px;
        }
        .scroll-table-wrapper.scroll-enabled {
            overflow-y: auto;
        }
        .scroll-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        .scroll-table thead th {
            background-color: #5e49e2;
            color: #ffffff;
            font-weight: 700;
            padding: 10px;
            border: 1px solid #d9dcee;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .scroll-table tbody td {
            color: #1f2937;
            padding: 8px 10px;
            border: 1px solid #e4e7f2;
        }
        .scroll-table tbody tr:nth-child(odd) {
            background-color: #f6f8ff;
        }
        .scroll-table tbody tr:nth-child(even) {
            background-color: #edf1ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    table_df = df.copy()

    # Keep non-default index values (e.g., classification labels) as a visible column.
    has_default_index = isinstance(table_df.index, pd.RangeIndex) and table_df.index.equals(
        pd.RangeIndex(start=0, stop=len(table_df), step=1)
    )
    if not has_default_index:
        index_col_name = table_df.index.name if table_df.index.name else "Label"
        table_df = table_df.reset_index().rename(columns={"index": index_col_name})

    numbered_df = table_df.copy()
    if show_serial_number:
        numbered_df.insert(0, "S/No", range(1, len(numbered_df) + 1))

    table_html = numbered_df.to_html(index=False, classes="scroll-table", border=0)
    wrapper_class = "scroll-table-wrapper scroll-enabled" if enable_vertical_scroll else "scroll-table-wrapper"
    wrapper_style = f'max-height:{table_height}px' if enable_vertical_scroll else ""
    st.markdown(
        f'<div class="{wrapper_class}" style="{wrapper_style}">{table_html}</div>',
        unsafe_allow_html=True,
    )


def prepare_features_for_prediction(test_data, scaler):
    """Prepare prediction features to match the scaler/model expected input shape."""
    if test_data is None:
        raise ValueError("No test data available.")

    # Remove known non-feature columns.
    feature_df = test_data.drop(columns=['Actual'], errors='ignore').copy()
    prediction_cols = [c for c in feature_df.columns if str(c).endswith('_Prediction')]
    if prediction_cols:
        feature_df = feature_df.drop(columns=prediction_cols)

    # Keep only numeric columns.
    feature_df = feature_df.select_dtypes(include=[np.number])

    if feature_df.empty:
        raise ValueError("No numeric feature columns found after removing label/prediction columns.")

    # If scaler is available, enforce exact feature count expected by training.
    if scaler is not None and hasattr(scaler, 'n_features_in_'):
        expected_features = int(scaler.n_features_in_)
        current_features = int(feature_df.shape[1])

        if current_features < expected_features:
            raise ValueError(
                f"Feature mismatch: expected {expected_features} features, got {current_features}."
            )
        if current_features > expected_features:
            feature_df = feature_df.iloc[:, :expected_features]

    return feature_df


def display_model_selection_and_prediction(models, scaler, test_data=None):
    """Display model selection and prediction interface"""
    st.subheader("🎯 Model Selection & Prediction")
    
    if not models:
        st.warning("No models loaded. Please train models first.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select a Model:",
        list(models.keys())
    )
    
    if selected_model:
        st.info(f"Selected Model: **{selected_model}**")
        
        # Display model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Type:**", type(models[selected_model]).__name__)
        
        with col2:
            if hasattr(models[selected_model], 'get_params'):
                st.write("**Hyperparameters:** Check model details")
        
        # Test data prediction
        if test_data is not None and 'Actual' in test_data.columns:
            st.subheader("Predictions on Test Data")

            try:
                X_test = prepare_features_for_prediction(test_data, scaler)
            except ValueError as exc:
                st.error(f"Cannot run prediction: {exc}")
                return

            # Scale test data
            if scaler:
                X_test_scaled = scaler.transform(X_test.values)
            else:
                X_test_scaled = X_test.values
            
            # Make predictions
            predictions = models[selected_model].predict(X_test_scaled)
            
            # Get probability predictions if available
            if hasattr(models[selected_model], 'predict_proba'):
                probabilities = models[selected_model].predict_proba(X_test_scaled)
            else:
                probabilities = None
            
            # Display results
            results_df = pd.DataFrame({
                'Actual': test_data['Actual'].values,
                'Predicted': predictions
            })
            
            if probabilities is not None:
                results_df['Confidence'] = probabilities.max(axis=1)
            
            display_scrollable_table(
                results_df,
                max_rows=TABLE_SCROLL_THRESHOLD_ROWS,
                show_serial_number=True,
            )
            
            # Calculate metrics
            
            accuracy = accuracy_score(test_data['Actual'].values, predictions)
            cm = confusion_matrix(test_data['Actual'].values, predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("\nAccuracy", f"{accuracy:.4f}")
            
            with col2:
                st.metric("Samples Predicted", len(predictions))
            
            # Confusion Matrix
            st.subheader("📈 Confusion Matrix")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(test_data['Actual'].values, predictions,
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            display_scrollable_table(
                report_df,
                max_rows=TABLE_SCROLL_THRESHOLD_ROWS,
                show_serial_number=False,
            )


def main():
    """Main Streamlit application"""
    
    # Auto-train models if they don't exist (needed for first Streamlit Cloud run)
    auto_train_models_if_missing()
    
    # Header introduction
    st.markdown("""
    <div class="dashboard-header" style="text-align: center;">
        <h1 style="color: #5e49e2; margin-bottom: 10px;">Welcome to the BITS ML Classification Dashboard</h1>
        <p style="color: #666; font-size: 16px;">Compare, analyze, and test multiple classification models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main title
    #st.title("🤖 ML Classification Models Dashboard")
    #st.markdown("---")
    
    # Load data and models
    models = load_models()
    scaler = load_scaler()
    results_df = load_results()
    test_data = load_test_data()
    
    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Predictions", "About"])
    
    with tab1:
        st.header("Model Performance Comparison")
        
        if results_df is not None:
            display_metrics_table(results_df)
        else:
            st.warning("No results found. Please train models first.")
    
    with tab2:
        st.header("Make Predictions")
        
        # Check if we have test data
        uploaded_data = display_dataset_upload()
        
        # Use uploaded data or default test data
        data_to_use = uploaded_data if uploaded_data is not None else test_data
        
        if models:
            display_model_selection_and_prediction(models, scaler, data_to_use)
        else:
            st.warning("No models loaded. Please train models first.")
    
    with tab3:
        st.header("ℹ️  About This Application")
        
        st.write("""
        ## ML Classification Models Comparison
        
        This interactive dashboard allows you to:
        - **Compare Performance**: View evaluation metrics for all 6 trained models
        - **Make Predictions**: Use trained models to make predictions on new data
        - **Analyze Results**: Visualize confusion matrices and classification reports
        
        ### Models Implemented:
        1. **Logistic Regression** - Linear classification model
        2. **Decision Tree** - Tree-based classification
        3. **K-Nearest Neighbor (kNN)** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        
        ### Evaluation Metrics:
        - **Accuracy** - Overall correctness of predictions
        - **AUC Score** - Area Under the ROC Curve
        - **Precision** - Correctly identified positive cases
        - **Recall** - True positive rate
        - **F1 Score** - Harmonic mean of precision and recall
        - **MCC** - Matthews Correlation Coefficient
        
        ### Dataset Information:
        - **Dataset**: Breast Cancer Classification
        - **Features**: 30 features
        - **Instances**: 569 samples
        - **Problem Type**: Binary Classification
        """)
        
        if results_df is not None:
            st.subheader("Model Summary")
            display_scrollable_table(results_df, max_rows=TABLE_SCROLL_THRESHOLD_ROWS, show_serial_number=False)


if __name__ == "__main__":
    main()
