# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# -----------------------------------------------------------------------------------------
# Section 2: Data Processing Class (Updated for File Upload)
# -----------------------------------------------------------------------------------------
class DataProcessor:
    def __init__(self):
        self.df = None

    @st.cache_data
    def load_data(_self, uploaded_file=None):
        """Load data from an uploaded file, a local file, or create sample data."""
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                _self.df = df
                # Success message moved to main function to avoid re-displaying on every page change
                return df
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                return None

        file_path = 'container_vm_sustainability_dataset.csv'
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                _self.df = df
                return df
            else:
                st.warning("⚠️ Default dataset not found. Generating sample data.")
                return _self.create_sample_data()
        except Exception as e:
            st.error(f"Error loading default data: {e}")
            return _self.create_sample_data()

    def create_sample_data(self):
        """Create sample data matching the dataset structure if the original file is missing."""
        np.random.seed(42)
        n_samples = 1000
        workload_types = ['web_service', 'microservice', 'database', 'ml_training', 'batch_processing']
        storage_types = ['ssd', 'hdd', 'nvme']
        datacenters = ['US-West', 'US-East', 'EU-Central', 'Asia-Pacific']
        data = {
            'deployment_type': np.random.choice(['container', 'vm'], n_samples, p=[0.6, 0.4]),
            'workload_type': np.random.choice(workload_types, n_samples),
            'cpu_cores': np.random.choice([1, 2, 4, 8, 16, 32], n_samples),
            'memory_gb': np.random.choice([2, 4, 8, 16, 32, 64], n_samples),
            'cpu_utilization': np.random.uniform(0.1, 0.95, n_samples),
            'memory_utilization': np.random.uniform(0.1, 0.9, n_samples),
            'storage_type': np.random.choice(storage_types, n_samples),
            'datacenter_location': np.random.choice(datacenters, n_samples),
        }
        self.df = pd.DataFrame(data)
        return self.df

    def apply_filters(self, df):
        """Apply filters to the dataframe based on sidebar inputs."""
        filtered_df = df.copy()
        
        st.sidebar.header("📊 Data Explorer Filters")
        deployment_types = st.sidebar.multiselect(
            "Deployment Types",
            options=df['deployment_type'].unique(),
            default=df['deployment_type'].unique()
        )
        filtered_df = filtered_df[filtered_df['deployment_type'].isin(deployment_types)]

        cpu_range = st.sidebar.slider(
            "CPU Cores Range",
            min_value=int(df['cpu_cores'].min()),
            max_value=int(df['cpu_cores'].max()),
            value=(int(df['cpu_cores'].min()), int(df['cpu_cores'].max()))
        )
        filtered_df = filtered_df[
            (filtered_df['cpu_cores'] >= cpu_range[0]) &
            (filtered_df['cpu_cores'] <= cpu_range[1])
        ]
        return filtered_df

# -----------------------------------------------------------------------------------------
# Section 3: Model Management Class (Corrected for Caching)
# -----------------------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.model = None
        self.feature_names = []

    def prepare_features(self, df):
        """Prepare features for the model, including one-hot encoding AND handling missing values."""
        feature_cols = ['workload_type', 'cpu_cores', 'memory_gb', 'cpu_utilization', 'memory_utilization', 'storage_type']
        
        # Check if all required columns exist in the dataframe
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Error: Your dataset is missing the following required columns: {', '.join(missing_cols)}")
            return None, None, None

        X = df[feature_cols].copy()
        y = df['deployment_type']
        
        X_encoded = pd.get_dummies(X, columns=['workload_type', 'storage_type'], drop_first=True)
        
        for col in X_encoded.columns:
            if X_encoded[col].isnull().any():
                X_encoded[col] = X_encoded[col].fillna(X_encoded[col].mean())
                
        feature_names = X_encoded.columns.tolist()
        return X_encoded, y, feature_names

    @st.cache_data
    def train_model(_self, df):
        """
        Train the Random Forest model.
        """
        X, y, feature_names = _self.prepare_features(df)
        
        if X is None: # Stop if feature preparation failed
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=model.classes_),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': feature_importance,
            'y_pred_proba': y_pred_proba,
            'classes': model.classes_
        }

    def predict(self, inputs):
        """Make a prediction for new inputs."""
        if self.model is None:
            st.error("Model is not available for prediction.")
            return None, None
        
        input_df = pd.DataFrame([inputs])
        input_encoded = pd.get_dummies(input_df)
        input_reindexed = input_encoded.reindex(columns=self.feature_names, fill_value=0)
        
        prediction = self.model.predict(input_reindexed)[0]
        probability = self.model.predict_proba(input_reindexed)[0]
        confidence = np.max(probability)
        
        return prediction, confidence

# -----------------------------------------------------------------------------------------
# Section 4: Charting and Visualization Class (Corrected)
# -----------------------------------------------------------------------------------------
class ChartManager:
    def plot_confusion_matrix(self, model_info):
        cm = model_info['confusion_matrix']
        labels = model_info['classes']
        fig = ff.create_annotated_heatmap(z=cm, x=list(labels), y=list(labels), colorscale='Blues', showscale=True)
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        return fig

    def plot_feature_importance(self, model_info):
        fig = px.bar(
            model_info['feature_importance'].head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importances'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig

    def plot_deployment_distribution(self, df):
        dist_data = df['deployment_type'].value_counts()
        fig = px.pie(values=dist_data.values, names=dist_data.index, title="Deployment Type Distribution")
        return fig

    def plot_geographic_distribution(self, df):
        # Added a check for the column's existence to prevent errors with diverse datasets
        if 'datacenter_location' in df.columns:
            geo_data = df['datacenter_location'].value_counts().reset_index()
            geo_data.columns = ['datacenter', 'count']
            fig = px.bar(geo_data, x='datacenter', y='count', title="Deployments by Datacenter Location")
            return fig
        return go.Figure().update_layout(title="Geographic data not available in this dataset.")


    def plot_confidence_distribution(self, model_info):
        confidences = np.max(model_info['y_pred_proba'], axis=1)
        fig = px.histogram(confidences, nbins=20, title="Prediction Confidence Distribution")
        fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
        return fig
    
    def plot_accuracy_breakdown(self, model_info):
        report = model_info['classification_report']
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        data = {'class': [], 'metric': [], 'score': []}
        for c in classes:
            for metric in ['precision', 'recall', 'f1-score']:
                data['class'].append(c.title())
                # --- FIX IS HERE ---
                # Changed .Title() to the correct .title()
                data['metric'].append(metric.title())
                # --- END OF FIX ---
                data['score'].append(report[c][metric])
        
        df = pd.DataFrame(data)
        fig = px.bar(df, x='class', y='score', color='metric', barmode='group', title="Performance Metrics by Class")
        return fig
# -----------------------------------------------------------------------------------------
# Section 5: UI Components Class (Corrected for Color)
# -----------------------------------------------------------------------------------------
class UIComponents:
    def render_header(self):
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center;">☁️ Cloud Deployment AI Classifier</h1>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### 🚀 Navigation")
            pages = ["🏠 Dashboard", "🔮 Make Prediction", "📊 Data Explorer", "🧠 Model Insights", "📈 Performance"]
            selected_page = st.radio("Choose a page:", pages, label_visibility="collapsed")
            st.markdown("---")
            
            st.markdown("""
            <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px;">
                <h4 style="color: #1f77b4;">🤖 AI Model Info</h4>
                <p style="color: #333; margin: 0;"><strong>Algorithm:</strong> Random Forest</p>
            </div>
            """, unsafe_allow_html=True)
            
            return selected_page

    def render_prediction_form(self):
        workload_types = ['web_service', 'microservice', 'database', 'ml_training', 'batch_processing']
        storage_types = ['ssd', 'hdd', 'nvme']
        
        with st.form("prediction_form"):
            st.subheader("🔧 Workload Configuration")
            col1, col2 = st.columns(2)
            with col1:
                workload_type = st.selectbox("Workload Type", workload_types)
                cpu_cores = st.select_slider("CPU Cores", [1, 2, 4, 8, 16, 32])
                memory_gb = st.select_slider("Memory (GB)", [2, 4, 8, 16, 32, 64])
            with col2:
                cpu_utilization = st.slider("Expected CPU Utilization (%)", 10, 95, 50) / 100.0
                memory_utilization = st.slider("Expected Memory Utilization (%)", 10, 95, 60) / 100.0
                storage_type = st.selectbox("Storage Type", storage_types)
            
            submitted = st.form_submit_button("🚀 Predict Deployment Type", use_container_width=True, type="primary")

        if submitted:
            return {
                'workload_type': workload_type, 'cpu_cores': cpu_cores, 'memory_gb': memory_gb,
                'cpu_utilization': cpu_utilization, 'memory_utilization': memory_utilization,
                'storage_type': storage_type
            }
        return None

    def render_prediction_results(self, prediction, confidence):
        if prediction == 'container':
            color, emoji, title = "#2ca02c", "📦", "Container"
            desc = "This workload is best suited for containerized deployment for its efficiency and scalability."
        else:
            color, emoji, title = "#ff7f0e", "🖥️", "Virtual Machine (VM)"
            desc = "This workload is best suited for a VM for better isolation and resource control."
        
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 1.5rem; text-align: center;">
            <h2 style="color: {color};">{emoji} Recommendation: {title}</h2>
            <p>{desc}</p>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------
# Section 6: Main Application Logic (Updated for File Upload)
# -----------------------------------------------------------------------------------------
def main():
    """Main application entry point."""
    st.set_page_config(page_title="Cloud Deployment AI Classifier", page_icon="☁️", layout="wide")

    # Initialize components
    data_processor = DataProcessor()
    model_manager = ModelManager()
    chart_manager = ChartManager()
    ui_components = UIComponents()
    
    ui_components.render_header()
    page = ui_components.render_sidebar()
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Custom Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        st.sidebar.success("Custom dataset loaded! ✅")

    df = data_processor.load_data(uploaded_file=uploaded_file)
    if df is None:
        st.error("❌ Data could not be loaded. Please upload a valid CSV or ensure the default dataset is available.")
        return
    
    model_info = model_manager.train_model(df)
    
    # Stop execution if the model training failed (e.g., due to missing columns)
    if model_info is None:
        return
        
    model_manager.model = model_info['model']
    model_manager.feature_names = model_info['feature_names']
    
    # Page routing
    if page == "🏠 Dashboard":
        render_dashboard(df, model_info, chart_manager)
    elif page == "🔮 Make Prediction":
        render_prediction_page(model_manager, ui_components)
    elif page == "📊 Data Explorer":
        render_data_explorer(df, data_processor, chart_manager)
    elif page == "🧠 Model Insights":
        render_model_insights(chart_manager, model_info)
    elif page == "📈 Performance":
        render_performance_page(chart_manager, model_info)

def render_dashboard(df, model_info, chart_manager):
    st.header("📊 Model Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Model Accuracy", f"{model_info['accuracy']:.1%}")
    col2.metric("📝 Total Records", f"{len(df):,}")
    col3.metric("📦 Container Share", f"{(df['deployment_type'] == 'container').mean():.1%}")
    col4.metric("🖥️ VM Share", f"{(df['deployment_type'] == 'vm').mean():.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_manager.plot_confusion_matrix(model_info), use_container_width=True)
        st.plotly_chart(chart_manager.plot_deployment_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(chart_manager.plot_feature_importance(model_info), use_container_width=True)
        st.plotly_chart(chart_manager.plot_geographic_distribution(df), use_container_width=True)

def render_prediction_page(model_manager, ui_components):
    st.header("🔮 Make a Prediction")
    st.markdown("Use our AI model to predict the optimal deployment type for your workload.")
    
    prediction_inputs = ui_components.render_prediction_form()
    
    if prediction_inputs:
        with st.spinner('🧠 Analyzing workload...'):
            prediction, confidence = model_manager.predict(prediction_inputs)
            if prediction is not None:
                ui_components.render_prediction_results(prediction, confidence)

def render_data_explorer(df, data_processor, chart_manager):
    st.header("📊 Data Explorer")
    st.markdown("Filter and explore the dataset used to train the model.")
    
    filtered_df = data_processor.apply_filters(df)
    st.dataframe(filtered_df, use_container_width=True)
    st.info(f"Showing **{len(filtered_df)}** of **{len(df)}** records.")

def render_model_insights(chart_manager, model_info):
    st.header("🧠 Model Insights")
    st.markdown("Understand which factors are most important for the model's decisions.")
    st.plotly_chart(chart_manager.plot_feature_importance(model_info), use_container_width=True)
    st.info("""
    **How to Read This Chart:** Features at the top are the most influential in the model's decision-making process. 
    For example, a high importance for `cpu_utilization` means the model relies heavily on this metric to differentiate 
    between workloads suitable for Containers vs. VMs.
    """)

def render_performance_page(chart_manager, model_info):
    st.header("📈 Detailed Model Performance")
    
    report = model_info['classification_report']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{model_info['accuracy']:.3f}")
    with col2:
        st.metric("Macro Avg F1-Score", f"{report['macro avg']['f1-score']:.3f}")
    with col3:
        st.metric("Weighted Avg F1-Score", f"{report['weighted avg']['f1-score']:.3f}")
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_manager.plot_accuracy_breakdown(model_info), use_container_width=True)
    with col2:
        st.plotly_chart(chart_manager.plot_confidence_distribution(model_info), use_container_width=True)

if __name__ == "__main__":
    main()