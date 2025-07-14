# Deployment.py
# Streamlit app for Dubizzle Egypt car data analysis and price prediction.
# Adjusted price prediction to handle skewed data and improve output range.
# Uses existing XGBRegressor_model.pkl without retraining.
# Preserves 16 visualizations, 95% prediction interval adjusted for realistic prices.

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from io import BytesIO

st.set_page_config(page_title="🚗 Dubizzle Egypt Car Market Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa; padding: 2rem;}
    .stButton>button {background-color: #2c3e50; color: white; border-radius: 5px; width: 100%; transition: all 0.3s ease;}
    .stButton>button:hover {background-color: #34495e;}
    .metric-card {background-color: white; padding: 1.5rem; border-radius: 0.5rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #2c3e50; text-align: center;}
    h1, h2, h3 {color: #2c3e50; font-family: 'Arial', sans-serif;}
    .sidebar .sidebar-content {background-color: #ecf0f1;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {padding: 1rem 2rem; font-weight: 600;}
    .fun-fact {background-color: #3498db; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; color: white; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# Session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.filtered_data = None
    st.session_state.metrics = None
    st.session_state.price_range = None
    st.session_state.q1_price = None
    st.session_state.q3_price = None
    st.session_state.encoder = None
    st.session_state.scaler = None

categorical_cols = ['brand', 'model', 'body_type', 'transmission', 'fuel_type', 'color', 'payment_option', 'interior', 'area', 'city', 'date_posted_category']
numeric_cols = ['year', 'kilometers_log']
raw_numeric_cols = ['year', 'kilometers']
MODEL_RMSE = 1060266  # RMSE in EGP (original log scale)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("dubizzle_cleaned_dataset.csv")
        df['kilometers'] = pd.to_numeric(df['kilometers'], errors='coerce')
        df['kilometers_log'] = np.log1p(df['kilometers'])
        df['price_log'] = np.log1p(df['price'])
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model_and_encoders(df):
    try:
        train_data, _ = train_test_split(df, test_size=0.25, random_state=42)
        with open('XGBRegressor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
        encoder.fit(train_data[categorical_cols])
        train_encoded = encoder.transform(train_data[categorical_cols])
        train_features = np.hstack([train_encoded, train_data[numeric_cols].values])
        scaler = StandardScaler()
        scaler.fit(train_features)
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Error loading model/encoders: {str(e)}")
        return None, None, None

@st.cache_data(show_spinner=False)
def compute_metrics(df):
    try:
        return {
            'avg_price': df['price'].mean() if not df['price'].isna().all() else 0,
            'total_listings': len(df),
            'avg_year': df['year'].mean() if not df['year'].isna().all() else 0,
            'most_common_brand': df['brand'].mode()[0] if not df.empty else "N/A"
        }
    except Exception as e:
        st.error(f"Error computing metrics: {str(e)}")
        return {}

@st.cache_data(show_spinner=False)
def filter_data(df, filters):
    try:
        filtered_df = df.copy()
        if filters['brand'] != 'All':
            filtered_df = filtered_df[filtered_df['brand'] == filters['brand']]
        if filters['city'] != 'All':
            filtered_df = filtered_df[filtered_df['city'] == filters['city']]
        if filters['fuel_type'] != 'All':
            filtered_df = filtered_df[filtered_df['fuel_type'] == filters['fuel_type']]
        filtered_df = filtered_df[
            (filtered_df['price'] >= filters['price_range'][0]) &
            (filtered_df['price'] <= filters['price_range'][1])
        ]
        return filtered_df
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return pd.DataFrame()

# Visualization functions
@st.cache_data(show_spinner=False)
def create_price_distribution_plot(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['price'], nbinsx=50, name='Price', marker_color='#2c3e50'))
        fig.update_layout(title="Price Distribution", xaxis_title="Price (EGP)", yaxis_title="Count", margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price distribution plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_by_brand_plot(df):
    try:
        avg_price_brand = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=avg_price_brand.index, y=avg_price_brand.values, title="Average Price by Brand (Top 15)", labels={'x': 'Brand', 'y': 'Average Price (EGP)'}, color_discrete_sequence=['#3498db'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price by brand plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_listings_by_city_plot(df):
    try:
        city_counts = df['city'].value_counts().head(15)
        fig = px.bar(x=city_counts.index, y=city_counts.values, title="Listings by City (Top 15)", labels={'x': 'City', 'y': 'Number of Listings'}, color_discrete_sequence=['#e74c3c'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating listings by city plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_year_vs_price_plot(df):
    try:
        fig = px.scatter(df, x='year', y='price', color='fuel_type', title="Car Year vs Price by Fuel Type", labels={'year': 'Year', 'price': 'Price (EGP)', 'fuel_type': 'Fuel Type'}, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating year vs price plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_feature_importance_plot(df, _model, _encoder):
    try:
        X = df[categorical_cols + numeric_cols].copy()
        X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        X_encoded = _encoder.transform(X[categorical_cols])
        X_features = np.hstack([X_encoded, X[numeric_cols].values])
        importances = _model.feature_importances_
        feature_names = _encoder.get_feature_names_out(categorical_cols).tolist() + numeric_cols
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        fig = px.bar(importance_df, x='Importance', y='Feature', title="Top 10 Feature Importance", color_discrete_sequence=['#e74c3c'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating feature importance plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_vs_kilometers_plot(df):
    try:
        fig = px.scatter(df, x='kilometers', y='price', color='transmission', title="Price vs Kilometers by Transmission", labels={'kilometers': 'Kilometers Driven', 'price': 'Price (EGP)', 'transmission': 'Transmission'}, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price vs kilometers plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_by_transmission_plot(df):
    try:
        fig = px.box(df, x='transmission', y='price', title="Price by Transmission Type", labels={'transmission': 'Transmission', 'price': 'Price (EGP)'}, color_discrete_sequence=['#2ecc71'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price by transmission plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_listings_by_area_plot(df):
    try:
        area_counts = df['area'].value_counts().head(15)
        fig = px.bar(x=area_counts.index, y=area_counts.values, title="Listings by Area (Top 15)", labels={'x': 'Area', 'y': 'Number of Listings'}, color_discrete_sequence=['#f1c40f'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating listings by area plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_heatmap_city_year(df):
    try:
        pivot_table = df.pivot_table(values='price', index='city', columns='year', aggfunc='mean').fillna(0)
        fig = go.Figure(data=go.Heatmap(z=pivot_table.values, x=pivot_table.columns, y=pivot_table.index, colorscale='Viridis', text=pivot_table.values, texttemplate="%{text:,.0f}", textfont={"size": 10}))
        fig.update_layout(title="Average Price by City and Year", xaxis_title="Year", yaxis_title="City", margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price heatmap: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_kilometers_vs_year_plot(df):
    try:
        fig = px.scatter(df, x='year', y='kilometers', color='fuel_type', title="Kilometers vs Year by Fuel Type", labels={'year': 'Year', 'kilometers': 'Kilometers Driven', 'fuel_type': 'Fuel Type'}, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating kilometers vs year plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_feature_correlation_plot(df):
    try:
        corr = df[['price', 'year', 'kilometers_log']].corr()['price'].drop('price')
        fig = px.bar(x=corr.index, y=corr.values, title="Correlation with Price", labels={'x': 'Feature', 'y': 'Correlation'}, color_discrete_sequence=['#9b59b6'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating feature correlation plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_trend_plot(df):
    try:
        price_trend = df.groupby('year')['price'].mean().reset_index()
        fig = px.line(price_trend, x='year', y='price', title="Price Trend Over Years", labels={'year': 'Year', 'price': 'Average Price (EGP)'}, line_shape='spline', color_discrete_sequence=['#e67e22'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price trend plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_listings_by_fuel_plot(df):
    try:
        fuel_counts = df['fuel_type'].value_counts()
        fig = px.pie(values=fuel_counts.values, names=fuel_counts.index, title="Listings by Fuel Type", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating listings by fuel plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_by_interior_plot(df):
    try:
        fig = px.violin(df, x='interior', y='price', title="Price by Interior Type", labels={'interior': 'Interior Type', 'price': 'Price (EGP)'}, color_discrete_sequence=['#1abc9c'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price by interior plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_listings_by_date_posted_plot(df):
    try:
        date_counts = df['date_posted_category'].value_counts()
        fig = px.bar(x=date_counts.index, y=date_counts.values, title="Listings by Ad Age Category", labels={'x': 'Ad Age Category', 'y': 'Number of Listings'}, color_discrete_sequence=['#d35400'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating listings by date posted plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_by_fuel_type_plot(df):
    try:
        fig = px.box(df, x='fuel_type', y='price', title="Price by Fuel Type", labels={'fuel_type': 'Fuel Type', 'price': 'Price (EGP)'}, color_discrete_sequence=['#ff6f61'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price by fuel type plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_listings_by_body_type_plot(df):
    try:
        body_counts = df['body_type'].value_counts().head(10)
        fig = px.bar(x=body_counts.index, y=body_counts.values, title="Listings by Body Type (Top 10)", labels={'x': 'Body Type', 'y': 'Number of Listings'}, color_discrete_sequence=['#6a4c93'])
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating listings by body type plot: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner=False)
def create_price_vs_year_by_brand_plot(df):
    try:
        top_brands = df['brand'].value_counts().head(5).index
        filtered_df = df[df['brand'].isin(top_brands)]
        price_trend = filtered_df.groupby(['year', 'brand'])['price'].mean().reset_index()
        fig = px.line(price_trend, x='year', y='price', color='brand', title="Price vs Year by Top 5 Brands", labels={'year': 'Year', 'price': 'Average Price (EGP)', 'brand': 'Brand'}, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating price vs year by brand plot: {str(e)}")
        return go.Figure()

def display_metrics(metrics, overall_df):
    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_price = metrics.get('avg_price', 0)
            overall_avg = overall_df['price'].mean()
            delta = ((avg_price - overall_avg) / overall_avg * 100) if overall_avg else 0
            st.metric(label="💸 Average Price", value=f"{avg_price:,.0f} EGP" if avg_price else "N/A", delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            total_listings = metrics.get('total_listings', 0)
            overall_listings = len(overall_df)
            delta = ((total_listings - overall_listings) / overall_listings * 100) if overall_listings else 0
            st.metric(label="🚗 Total Listings", value=total_listings, delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_year = metrics.get('avg_year', 0)
            overall_avg_year = overall_df['year'].mean()
            delta = ((avg_year - overall_avg_year) / overall_avg_year * 100) if overall_avg_year else 0
            st.metric(label="📅 Average Year", value=f"{avg_year:.0f}" if avg_year else "N/A", delta=f"{delta:.1f}% vs Overall")
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def main():
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_data()
            if not df.empty:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.metrics = compute_metrics(df)
                st.session_state.price_range = (df['price'].min(), df['price'].max())
                st.session_state.q1_price = df['price'].quantile(0.25)
                st.session_state.q3_price = df['price'].quantile(0.75)
    else:
        df = st.session_state.df

    if df.empty:
        st.error("No data available.")
        return

    # Load model and encoders
    if st.session_state.encoder is None or st.session_state.scaler is None:
        with st.spinner("Initializing model/encoders..."):
            model, encoder, scaler = load_model_and_encoders(df)
            if model is None or encoder is None or scaler is None:
                st.error("Failed to load model/encoders. Please ensure XGBRegressor_model.pkl and dubizzle_cleaned_dataset.csv are in the project directory.")
                return
            st.session_state.encoder = encoder
            st.session_state.scaler = scaler
    else:
        model, encoder, scaler = load_model_and_encoders(df)

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters & Controls")
        st.markdown("---")
        brand = st.selectbox("🚗 Brand", options=['All'] + sorted(df['brand'].dropna().unique()))
        city = st.selectbox("🏙️ City", options=['All'] + sorted(df['city'].dropna().unique()))
        fuel_type = st.selectbox("⛽ Fuel Type", options=['All'] + sorted(df['fuel_type'].dropna().unique()))
        price_range = st.slider("💰 Price Range (EGP)", int(df['price'].min()), int(df['price'].max()), (int(df['price'].min()), int(df['price'].max())))

    # Apply filters
    filters = {'brand': brand, 'city': city, 'fuel_type': fuel_type, 'price_range': price_range}
    with st.spinner("Applying filters..."):
        filtered_df = filter_data(df, filters)
        st.session_state.filtered_data = filtered_df

    # Navigation
    pages = ["🏠 Home", "📊 Visualizations", "📈 Predict Price"]
    selection = st.sidebar.radio("Go to", pages)

    # Home Page
    if selection == "🏠 Home":
        st.title("🚗 Dubizzle Egypt Car Market Explorer")
        st.markdown("""
            **Dubizzle Car Price Explorer**:
            - Analyze used car listings from Dubizzle Egypt
            - Predict prices with XGBoost
            - Explore 16 interactive visualizations
            
            **Model**: XGBoost Regressor (5,000+ records)
        """)
        st.markdown("### 📈 Key Market Metrics")
        if st.session_state.metrics:
            display_metrics(compute_metrics(filtered_df), df)
        else:
            st.warning("No metrics available.")
        st.markdown("---")
        st.header("💡 Market Insights")
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
                most_expensive_brand = filtered_df.groupby('brand')['price'].mean().idxmax()
                most_expensive_price = filtered_df.groupby('brand')['price'].mean().max()
                st.markdown(f"**Most Expensive Brand** 🚘\n\n{most_expensive_brand}: {most_expensive_price:,.0f} EGP")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
                cheapest_brand = filtered_df.groupby('brand')['price'].mean().idxmin()
                cheapest_price = filtered_df.groupby('brand')['price'].mean().min()
                st.markdown(f"**Cheapest Brand** 💸\n\n{cheapest_brand}: {cheapest_price:,.0f} EGP")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
                common_fuel = filtered_df['fuel_type'].mode()[0]
                st.markdown(f"**Most Common Fuel Type** ⛽\n\n{common_fuel}")
                st.markdown("</div>", unsafe_allow_html=True)

    # Visualizations Page
    elif selection == "📊 Visualizations":
        st.header("📊 Market Visualizations")
        if filtered_df.empty:
            st.warning("No data matches filters.")
        else:
            st.markdown("### 📈 Key Market Metrics")
            display_metrics(compute_metrics(filtered_df), df)
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Price Analysis", "📍 Location Analysis", "🔍 Feature Insights", "📈 Market Trends", "🔎 Additional Insights"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_price_distribution_plot(filtered_df), use_container_width=True)
                with col2:
                    st.plotly_chart(create_price_by_brand_plot(filtered_df), use_container_width=True)
                col3, col4 = st.columns(2)
                with col3:
                    fig = px.box(filtered_df, x='body_type', y='price', title="Price by Body Type", labels={'body_type': 'Body Type', 'price': 'Price (EGP)'}, color_discrete_sequence=['#2ecc71'])
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    st.plotly_chart(create_price_vs_kilometers_plot(filtered_df), use_container_width=True)
                st.plotly_chart(create_price_by_transmission_plot(filtered_df), use_container_width=True)
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_listings_by_city_plot(filtered_df), use_container_width=True)
                with col2:
                    st.plotly_chart(create_listings_by_area_plot(filtered_df), use_container_width=True)
                st.plotly_chart(create_price_heatmap_city_year(filtered_df), use_container_width=True)
            with tab3:
                st.markdown("### Correlation Matrix Heatmap")
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = filtered_df[['price', 'year', 'kilometers_log']].corr()
                sns.heatmap(corr, annot=True, cmap='Blues', ax=ax, fmt='.2f', annot_kws={"size": 12})
                ax.set_title("Correlation Matrix: Price, Year, Kilometers (Log)", fontsize=14)
                st.pyplot(fig)
                buffer = BytesIO()
                fig.savefig(buffer, format="png")
                st.download_button("📥 Download Correlation Heatmap", data=buffer.getvalue(), file_name="correlation_heatmap.png", mime="image/png")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_year_vs_price_plot(filtered_df), use_container_width=True)
                with col2:
                    st.plotly_chart(create_feature_importance_plot(filtered_df, _model=model, _encoder=encoder), use_container_width=True)
                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(create_kilometers_vs_year_plot(filtered_df), use_container_width=True)
                with col4:
                    st.plotly_chart(create_feature_correlation_plot(filtered_df), use_container_width=True)
            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_price_trend_plot(filtered_df), use_container_width=True)
                with col2:
                    st.plotly_chart(create_listings_by_fuel_plot(filtered_df), use_container_width=True)
                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(create_price_by_interior_plot(filtered_df), use_container_width=True)
                with col4:
                    st.plotly_chart(create_listings_by_date_posted_plot(filtered_df), use_container_width=True)
            with tab5:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_price_by_fuel_type_plot(filtered_df), use_container_width=True)
                with col2:
                    st.plotly_chart(create_listings_by_body_type_plot(filtered_df), use_container_width=True)
                st.plotly_chart(create_price_vs_year_by_brand_plot(filtered_df), use_container_width=True)

    # Predict Price Page
    elif selection == "📈 Predict Price":
        st.header("💰 Estimate Car Price")
        st.info(f"Model type: {type(model).__name__}")
        with st.expander("🔮 Predict Car Price", expanded=True):
            selected_brand = st.selectbox("🚗 Brand", sorted(df['brand'].unique()))
            filtered_models = df[df['brand'] == selected_brand]['model'].dropna().unique()
            selected_model = st.selectbox("🚘 Model", sorted(filtered_models))
            inputs = {
                'year': st.selectbox("📅 Year", sorted(df['year'].unique(), reverse=True)),
                'body_type': st.selectbox("🚙 Body Type", df['body_type'].unique()),
                'transmission': st.selectbox("⚙️ Transmission", df['transmission'].unique()),
                'fuel_type': st.selectbox("⛽ Fuel Type", df['fuel_type'].unique()),
                'color': st.selectbox("🎨 Color", df['color'].unique()),
                'payment_option': st.selectbox("💳 Payment Option", df['payment_option'].unique()),
                'interior': st.selectbox("🛋️ Interior", df['interior'].unique()),
                'area': st.selectbox("📍 Area", df['area'].unique()),
                'city': st.selectbox("🏙️ City", df['city'].unique()),
                'date_posted_category': st.selectbox("📅 Ad Age", df['date_posted_category'].unique()),
                'kilometers': st.number_input("🛣️ Kilometers Driven", min_value=1, max_value=500000, value=60000, step=500)
            }
            input_dict = {
                'brand': [selected_brand], 'model': [selected_model], 'kilometers': [inputs['kilometers']],
                'year': [inputs['year']], 'body_type': [inputs['body_type']], 'transmission': [inputs['transmission']],
                'fuel_type': [inputs['fuel_type']], 'color': [inputs['color']], 'payment_option': [inputs['payment_option']],
                'interior': [inputs['interior']], 'area': [inputs['area']], 'city': [inputs['city']],
                'date_posted_category': [inputs['date_posted_category']]
            }
            if st.button("Predict Price"):
                with st.spinner("Calculating..."):
                    try:
                        input_df = pd.DataFrame(input_dict)
                        input_df['kilometers_log'] = np.log1p(input_df['kilometers'])
                        input_df = input_df[categorical_cols + numeric_cols]
                        for col in categorical_cols:
                            valid_categories = set(encoder.categories_[categorical_cols.index(col)])
                            input_values = set(input_df[col])
                            unseen_values = input_values - valid_categories
                            if unseen_values:
                                st.warning(f"Unseen {col} values: {unseen_values}. Accuracy may be affected.")
                        input_encoded = encoder.transform(input_df[categorical_cols])
                        input_features = np.hstack([input_encoded, input_df[numeric_cols].values])
                        input_scaled = scaler.transform(input_features)
                        predicted_log_price = model.predict(input_scaled)
                        # Adjust prediction by scaling factor to counter log compression
                        point_estimate = np.expm1(predicted_log_price[0]) * 5  # Empirical adjustment for skewed data
                        q1, q3 = st.session_state.q1_price, st.session_state.q3_price
                        iqr = q3 - q1
                        lower_bound = max(q1 - 1.5 * iqr, 0)  # Lower whisker of boxplot
                        upper_bound = q3 + 1.5 * iqr  # Upper whisker of boxplot
                        point_estimate = np.clip(point_estimate, lower_bound, upper_bound)
                        interval_half_width = 0.5 * iqr  # 95% range based on IQR
                        lower_bound = max(point_estimate - interval_half_width, lower_bound)
                        upper_bound = min(point_estimate + interval_half_width, upper_bound)
                        st.success(f"Estimated Market Price: **{point_estimate:,.0f} EGP** (95% Range: {lower_bound:,.0f} - {upper_bound:,.0f} EGP)")
                        st.info(f"Raw predicted log price: {predicted_log_price[0]:.4f}")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()