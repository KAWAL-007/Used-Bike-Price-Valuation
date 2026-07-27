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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Used Bike Price Valuation",
    page_icon="🏍️",
    layout="wide"
)

MODEL_FILE = "bike_price_model.pkl"
DATA_FILE = "Used_Bikes.csv"

# ==========================================
# 2. AUTOMATED MODEL TRAINING & PIPELINE
# ==========================================
@st.cache_resource
def get_or_train_model():
    """Trains the model if pickle file doesn't exist, otherwise loads it."""
    if not os.path.exists(MODEL_FILE):
        if not os.path.exists(DATA_FILE):
            st.error(f"Error: Dataset '{DATA_FILE}' not found! Please place it in the working directory.")
            st.stop()

        st.info("⚡ Training model pipeline for the first time... Please wait.")
        
        # Load & Clean Data
        df = pd.read_csv("Used_Bikes.csv")
        df.drop_duplicates(inplace=True)
        
        # Drop bike_name column if present
        if 'bike_name' in df.columns:
            df.drop(columns=['bike_name'], inplace=True)

        X = df.drop(columns=['price'])
        y = df['price']

        categorical_cols = ['city', 'owner', 'brand']
        numerical_cols = ['kms_driven', 'age', 'power']

        # Column Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        # Full ML Pipeline
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Train Split & Model Fitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_pipeline.fit(X_train, y_train)

        # Export Package
        model_payload = {
            'model': rf_pipeline,
            'cities': sorted(df['city'].unique().tolist()),
            'brands': sorted(df['brand'].unique().tolist()),
            'owners': list(df['owner'].unique())
        }

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_payload, f)

    # Load artifacts
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

# Initialize Model Artifacts
artifacts = get_or_train_model()
model = artifacts['model']
cities = artifacts['cities']
brands = artifacts['brands']
owners = artifacts['owners']

# Dynamic Dataset Loader for EDA
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
st.write("Predict pre-owned motorcycle market prices and explore dataset insights.")

# Sidebar Navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select View", ["Price Valuation Tool", "Market EDA Analytics"])

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

    st.markdown("---")
    
    if st.button("💰 Estimate Market Price", use_container_width=True):
        input_data = pd.DataFrame({
            'city': [city],
            'kms_driven': [kms_driven],
            'owner': [owner],
            'age': [age],
            'power': [power],
            'brand': [brand]
        })
        
        predicted_val = model.predict(input_data)[0]
        
        st.success(f"### Estimated Price: ₹{predicted_val:,.2f}")
        
        st.metric(
            label="Predicted Valuation", 
            value=f"₹{predicted_val:,.0f}",
            delta=f"{power} cc | {age} yrs old"
        )

# ------------------------------------------
# PAGE 2: MARKET EDA ANALYTICS
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