import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from src.feature_engineering import create_features

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# ======================
# LOAD MODELS
# ======================
clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")

# ======================
# TITLE
# ======================
st.title("🏡 Real Estate Investment Advisor")

# =====================================================
# SIDEBAR INPUT UI
# =====================================================
st.sidebar.header("📥 Property Inputs")

city = st.sidebar.selectbox(
    "City",
    ["Mumbai", "Delhi", "Bangalore", "Chennai"],
    key="city_select"
)

property_type = st.sidebar.selectbox(
    "Property Type",
    ["Apartment", "Independent House", "Villa"],
    key="property_type_select"
)

transport = st.sidebar.selectbox(
    "Transport",
    ["High", "Medium", "Low"],
    key="transport_select"
)

bhk = st.sidebar.slider(
    "BHK",
    1,
    6,
    2,
    key="bhk_slider"
)

sqft = st.sidebar.number_input(
    "Size (SqFt)",
    500,
    5000,
    1200,
    key="sqft_input"
)

price = st.sidebar.number_input(
    "Price (Lakhs)",
    10,
    1000,
    50,
    key="price_input"
)

year = st.sidebar.slider(
    "Year Built",
    1980,
    2025,
    2010,
    key="year_slider"
)

schools = st.sidebar.slider(
    "Schools Nearby",
    0,
    10,
    2,
    key="schools_slider"
)

hospitals = st.sidebar.slider(
    "Hospitals Nearby",
    0,
    10,
    1,
    key="hospitals_slider"
)

# ======================
# INPUT DATAFRAME
# ======================
input_df = pd.DataFrame([{
    "City": city,
    "State": "Unknown",
    "Locality": "Unknown",
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": sqft,
    "Price_in_Lakhs": price,
    "Year_Built": year,
    "Furnished_Status": "Semi",
    "Floor_No": 1,
    "Total_Floors": 5,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hospitals,
    "Public_Transport_Accessibility": transport,
    "Parking_Space": "Yes",
    "Security": "Yes",
    "Amenities": "Basic",
    "Facing": "North",
    "Owner_Type": "Individual",
    "Availability_Status": "Available"
}])

# ======================
# FEATURE ENGINEERING
# ======================
input_df = create_features(input_df)

input_df = input_df.drop(
    columns=["Good_Investment", "Future_Price"],
    errors="ignore"
)

input_df["ID"] = 0

# ======================
# PREDICTIONS
# ======================
if st.sidebar.button("🔍 Predict", key="predict_btn"):

    pred = clf.predict(input_df)[0]

    prob = clf.predict_proba(input_df)[0][1]

    future_price = reg.predict(input_df)[0]

    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Investment",
            "Good ✅" if pred == 1 else "Bad ❌"
        )

    with col2:
        st.metric(
            "Confidence",
            f"{prob:.2f}"
        )

    with col3:
        st.metric(
            "Future Price (Lakhs)",
            f"{future_price:.2f}"
        )

    # ======================
    # SIMPLE INSIGHT BOX
    # ======================
    st.subheader("💡 Quick Insight")

    if sqft > 2000:
        st.info(
            "Large property → higher appreciation potential"
        )

    if transport == "High":
        st.success(
            "Good transport connectivity increases demand"
        )

    if schools > 5:
        st.success(
            "High education proximity boosts property value"
        )

# =====================================================
# VISUAL INSIGHTS (EDA STYLE SECTION)
# =====================================================

st.markdown("---")

st.subheader(
    "📈 Market Insights Dashboard",
    anchor="market_insights_dashboard"
)

# =====================================================
# SAMPLE SYNTHETIC DATASET
# =====================================================
df_vis = pd.DataFrame({
    "City": ["Mumbai", "Delhi", "Bangalore", "Chennai"] * 25,
    "Price_in_Lakhs": np.random.randint(30, 300, 100),
    "Size_in_SqFt": np.random.randint(500, 4000, 100),
    "BHK": np.random.randint(1, 5, 100)
})

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 City Prices",
    "🏠 Size vs Price",
    "🏘 BHK Distribution",
    "🔥 Correlation"
])

# =====================================================
# 1. CITY PRICE TAB
# =====================================================
with tab1:

    city_avg = (
        df_vis.groupby("City")["Price_in_Lakhs"]
        .mean()
        .reset_index()
    )

    fig1 = px.bar(
        city_avg,
        x="City",
        y="Price_in_Lakhs",
        title="📍 City-wise Average Price",
        color="Price_in_Lakhs"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        key="city_price_chart"
    )

    st.info(
        "📌 Delhi has the highest average property price, "
        "while Chennai is the most affordable among the four cities."
    )

# =====================================================
# 2. SIZE VS PRICE TAB
# =====================================================
with tab2:

    fig2 = px.scatter(
        df_vis,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        color="BHK",
        title="🏠 Size vs Price Relationship"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
        key="size_vs_price_chart"
    )

    st.info(
        "📌 Property prices generally increase with size, "
        "but there is noticeable variation indicating "
        "other factors also influence pricing."
    )

# =====================================================
# 3. BHK DISTRIBUTION TAB
# =====================================================
with tab3:

    fig3 = px.box(
        df_vis,
        x="BHK",
        y="Price_in_Lakhs",
        title="🏘 BHK vs Price Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
        key="bhk_distribution_chart"
    )

    st.info(
        "📌 Higher BHK homes tend to have higher median prices, "
        "with 4 BHK properties showing the strongest overall price range."
    )

# =====================================================
# 4. CORRELATION TAB
# =====================================================
with tab4:

    st.subheader(
        "🔥 Feature Correlation Heatmap",
        anchor="correlation_heatmap"
    )

    corr = df_vis.corr(numeric_only=True)

    fig4, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(
        fig4,
        clear_figure=True
    )

    st.info(
        "📌 Price has a weak positive correlation with "
        "both size and BHK, while size and BHK themselves "
        "are only minimally related."
    )