import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Machine Learning Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Used Bike Valuation & Analytics",
    page_icon="🏍️",
    layout="wide"
)

MODEL_FILE = "bike_price_models.pkl"
DATA_FILE = "Used_Bikes.csv"

# ==========================================
# 2. MODEL TRAINING & PIPELINE FUNCTION
# ==========================================
@st.cache_resource
def get_or_train_models():
    """Trains multiple ML models if pickle file doesn't exist, otherwise loads it."""
    if not os.path.exists(MODEL_FILE):
        if not os.path.exists(DATA_FILE):
            st.error(f"Error: Dataset '{DATA_FILE}' not found! Please place it in the working directory.")
            st.stop()

        st.info("⚡ Training ML model suite for the first time... Please wait.")
        
        # Load & Clean Data
        df = pd.read_csv(DATA_FILE)
        df.drop_duplicates(inplace=True)
        
        if 'bike_name' in df.columns:
            df.drop(columns=['bike_name'], inplace=True)

        X = df.drop(columns=['price'])
        y = df['price']

        categorical_cols = ['city', 'owner', 'brand']
        numerical_cols = ['kms_driven', 'age', 'power']

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Dictionary of Candidate Models
        candidate_models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Linear Regression": LinearRegression()
        }

        trained_models = {}
        metrics_summary = []

        # Train and Evaluate Each Model
        for name, algo in candidate_models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', algo)
            ])
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            trained_models[name] = pipeline
            metrics_summary.append({
                'Model': name,
                'R2 Score': round(r2, 4),
                'MAE (₹)': round(mae, 2),
                'RMSE (₹)': round(rmse, 2)
            })

        metrics_df = pd.DataFrame(metrics_summary).sort_values(by='R2 Score', ascending=False)

        # Package Artifacts
        model_payload = {
            'models': trained_models,
            'metrics': metrics_df,
            'cities': sorted(df['city'].unique().tolist()),
            'brands': sorted(df['brand'].unique().tolist()),
            'owners': list(df['owner'].unique())
        }

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_payload, f)

    # Load Artifacts
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

# Load Artifacts
artifacts = get_or_train_models()
models = artifacts['models']
metrics_df = artifacts['metrics']
cities = artifacts['cities']
brands = artifacts['brands']
owners = artifacts['owners']

@st.cache_data
def load_raw_dataset():
    if os.path.exists(DATA_FILE):
        df_raw = pd.read_csv(DATA_FILE)
        df_raw.drop_duplicates(inplace=True)
        return df_raw
    return None

# ==========================================
# 3. STREAMLIT FRONTEND INTERFACE
# ==========================================
st.title("🏍️ Used Bike Valuation & Analytics Dashboard")

# Sidebar Navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select View", [
    "Price Valuation Tool", 
    "Model Performance Comparison", 
    "Market EDA Analytics"
])

# ------------------------------------------
# PAGE 1: PRICE VALUATION TOOL
# ------------------------------------------
if app_mode == "Price Valuation Tool":
    st.subheader("📋 Enter Motorcycle Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand = st.selectbox("Select Brand", brands)
        owner = st.selectbox("Owner Sequence", owners)
        
    with col2:
        city = st.selectbox("Select City Location", cities)
        age = st.number_input("Bike Age (Years)", min_value=1, max_value=40, value=4, step=1)
        
    with col3:
        power = st.number_input("Engine Capacity (cc)", min_value=80, max_value=2000, value=150, step=10)
        kms_driven = st.number_input("Total Kilometers Driven", min_value=1, max_value=500000, value=15000, step=1000)

    # Model Selection Dropdown
    st.markdown("---")
    st.subheader("⚙️ Select Machine Learning Algorithm")
    selected_model_name = st.selectbox("Choose Model for Prediction", list(models.keys()))

    if st.button("💰 Estimate Market Price", use_container_width=True):
        input_data = pd.DataFrame({
            'city': [city],
            'kms_driven': [kms_driven],
            'owner': [owner],
            'age': [age],
            'power': [power],
            'brand': [brand]
        })
        
        chosen_model = models[selected_model_name]
        predicted_val = chosen_model.predict(input_data)[0]
        
        st.success(f"### Estimated Price: ₹{predicted_val:,.2f}")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Predicted Valuation", value=f"₹{predicted_val:,.0f}")
        with col_m2:
            st.metric(label="Algorithm Used", value=selected_model_name)

# ------------------------------------------
# PAGE 2: MODEL PERFORMANCE COMPARISON
# ------------------------------------------
elif app_mode == "Model Performance Comparison":
    st.subheader("📊 Machine Learning Model Benchmarks")
    st.write("Compare predictive performance across multiple algorithms trained on the dataset.")

    st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### $R^2$ Score Comparison (Higher is better)")
        fig_r2 = px.bar(
            metrics_df, 
            x='Model', 
            y='R2 Score', 
            color='R2 Score',
            text='R2 Score',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        st.markdown("#### Mean Absolute Error (MAE in ₹) (Lower is better)")
        fig_mae = px.bar(
            metrics_df, 
            x='Model', 
            y='MAE (₹)', 
            color='MAE (₹)',
            text='MAE (₹)',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_mae, use_container_width=True)

# ------------------------------------------
# PAGE 3: MARKET EDA ANALYTICS
# ------------------------------------------
elif app_mode == "Market EDA Analytics":
    st.subheader("📊 Pre-Owned Market EDA Insights")
    
    df_raw = load_raw_dataset()
    
    if df_raw is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 Listing Volume by Brand")
            brand_counts = df_raw['brand'].value_counts().head(10).reset_index()
            brand_counts.columns = ['brand', 'count']
            fig1 = px.bar(brand_counts, x='brand', y='count', color='count', color_continuous_scale='Viridis')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("#### Engine Power vs Price Distribution")
            fig2 = px.scatter(
                df_raw, 
                x='power', 
                y='price', 
                color='owner', 
                hover_data=['brand', 'age'],
                opacity=0.6
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Price Depreciation Across Vehicle Age")
        fig3 = px.box(df_raw, x='age', y='price', color='owner')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Dataset unavailable for EDA display.")
