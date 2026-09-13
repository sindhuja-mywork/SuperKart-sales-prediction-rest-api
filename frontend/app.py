
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Sets the page layout to centred mode and adds a title
st.set_page_config(page_title="SuperKart Sales Prediction Platform", layout="centered")

# Injects custom CSS to style
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 2rem;
        font-weight: 700;
        color: #7c3aed;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.1rem;
        color: #4a5568;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Feature boxes styling */
    .feature-box {
        text-align: center;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.25rem 0;
        margin-bottom: 0.2rem;
    }
    .feature-title {
        font-weight: 600;
        color: #7c3aed;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .feature-desc {
        color: #7c3aed;
        font-size: 0.85rem;
        margin: 0;
    }

    /* Input section styling */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #718096;
        margin-bottom: 2rem;
    }

    /* Streamlit button customization */
     div.stButton {
        text-align: center;
        display: flex;
        justify-content: center;
    }

    .stButton button {
        background-color: #7c3aed;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        max-width: 300px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
    <div class="main-header">
        <div class="main-title">Unlock Your Sales Potential 🚀</div>
        <div class="sub-title">SuperKart Predict!</div>
        <p class="description">
            Predict smarter, sell better! Our AI-powered platform delivers instant sales
            forecasts to help you optimize inventory, understand customer demand, and
            maximize profits with confidence.
        </p>
    </div>
""", unsafe_allow_html=True)

# Creates two columns to display platform features
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🧠 Regression Model</div>
            <div class="feature-desc">Random Forest</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-box">
            <div class="feature-title">⭐ High Performance Model</div>
            <div class="feature-desc">with 91% R² Score</div>
        </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown('<div class="section-title">Enter your data to get instant predictions</div>', unsafe_allow_html=True)

# 2-column layout for input fields
col1, col2 = st.columns(2)

with col1:
    Product_Weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.66,
        help="Weight of the product (numerical value)",
    )

    Product_Sugar_Content = st.selectbox(
        "Product Sugar Content",
        ["Low Sugar", "Regular", "No Sugar"]
    )

    Product_Allocated_Area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.068,
        help="Ratio of the allocated display area of each product to the total display area of all the products in a store",
    )

    Product_MRP = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=116.7,
        help="Maximum retail price of each product (numerical value)",
    )

    Store_Size = st.selectbox(
        "Store Size",
        [ "Small", "Medium", "High"],
    )
with col2:

        Store_Location_City_Type = st.selectbox(
            "Store Location City Type",
            ["Tier 1", "Tier 2", "Tier 3"]
        )

        Store_Type = st.selectbox(
            "Store Type",
            ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
        )

        Store_Age_Years = st.number_input(
            "Store Age (Years)",
            min_value=0,
            value=17,
            help="Age of the store")

        Product_Type_Category = st.selectbox(
            "Product Type Category",
            ["Perishables", "Non Perishables"]
        )

        Product_Id_char = st.selectbox(
            "Product Id Char",
            ["FD", "NC", "DR"]
        )

st.divider()

# Button for running predictions
if st.button("⚡ Run Prediction", type="primary", use_container_width=True):

    if Product_Weight == 0.0:
        st.warning("⚠️ Please enter a valid Product Weight")
    elif Product_MRP == 0.0:
        st.warning("⚠️ Please enter a valid Product MRP")
    else:
        # Prepare data dictionary for API request
        product_data = {
            "Product_Weight": Product_Weight,
            "Product_Sugar_Content": Product_Sugar_Content,
            "Product_Allocated_Area": Product_Allocated_Area,
            "Product_MRP": Product_MRP,
            "Store_Size": Store_Size,
            "Store_Location_City_Type": Store_Location_City_Type,
            "Store_Type": Store_Type,
            "Store_Age_Years": Store_Age_Years,
            "Product_Type_Category": Product_Type_Category,
            "Product_Id_char": Product_Id_char
        }

        # Spinner to show animation during API call
        with st.spinner("Running prediction..."):
            try:
                # API call to get prediction
                response = requests.post(
                    f"{BACKEND_URL}/v1/predict",
                    json=product_data,
                    headers={
                        "Content-Type": "application/json"
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    predicted_sales = result.get("Sales", 0)

                    # Displays prediction results
                    st.success("✅ Prediction Complete!")
                    st.metric(
                            label="Predicted Sales",
                            value=f"£{predicted_sales:.2f}"
                        )
                else:
                    # Error if API call fails
                    st.error(f"❌ Error in API request: {response.status_code}")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
