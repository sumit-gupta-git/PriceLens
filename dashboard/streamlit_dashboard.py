# streamlit_app_updated.py
import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np

BACKEND_URL = "http://localhost:5000/predict"
SAMPLE_DATA_URL = "http://localhost:5000/sample-data"

st.set_page_config(page_title="PriceLens Dashboard", layout="wide")

# ------------- Sidebar Inputs -------------------
st.sidebar.header("Enter Car Features")
brand = st.sidebar.selectbox("Brand", [
    "Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Mahindra", "Renault", "Nissan", "Tata", "Volkswagen", "BMW", "Mercedes", "Audi", "Other"
])
vehicle_age = st.sidebar.slider("Vehicle Age (Years)", 0, 25, 5)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=40000, step=500)
seller_type = st.sidebar.radio("Seller Type", ["Dealer", "Individual", "TrustMark"])
fuel_type = st.sidebar.radio("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission_type = st.sidebar.radio("Transmission", ["Manual", "Automatic"])
mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
engine = st.sidebar.number_input("Engine (CC)", min_value=600, max_value=6000, value=1200, step=50)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=80.0, step=1.0)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8])

if st.sidebar.button("Predict Price"):
    payload = {
        "brand": brand,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }
    with st.spinner("Predicting..."):
        try:
            resp = requests.post(BACKEND_URL, json=payload)
            if resp.status_code == 200:
                price = resp.json()["predicted_price"]
                st.success(f"Estimated Price: ₹{price:,.2f}")
            else:
                st.error(f"Prediction error: {resp.text}")
        except Exception as e:
            st.error(f"API call failed: {e}")

# ---------------- Dashboard Analytics ----------------------
st.title("Used Car Market Analytics Dashboard")
try:
    resp = requests.get(SAMPLE_DATA_URL)
    if resp.status_code == 200:
        market = resp.json()['data']
    else:
        market = None
except Exception:
    market = None

st.markdown(":blue[The following visualizations help you understand Indian used-car market trends.]")

if market:
    # Bar: Average price by brand
    st.subheader("Average Resale Price by Brand")
    df_brand = pd.DataFrame(market['brand_stats'])
    bar = alt.Chart(df_brand).mark_bar().encode(
        x=alt.X('brand:N', sort='-y'),
        y=alt.Y('avg_price:Q', title='Avg Price (INR)'),
        color=alt.value('#4069E5'),
        tooltip=['brand', 'avg_price', 'count']
    )
    st.altair_chart(bar, use_container_width=True)

    # Line: Avg price by vehicle age
    st.subheader("Average Price vs Vehicle Age")
    df_age = pd.DataFrame(market['age_price_data'])
    line = alt.Chart(df_age).mark_line(point=True).encode(
        x=alt.X('vehicle_age:O', title='Vehicle Age (Years)'),
        y=alt.Y('avg_price:Q', title='Avg Price (INR)'),
        tooltip=['vehicle_age', 'avg_price', 'count']
    )
    st.altair_chart(line, use_container_width=True)

    # Boxplot: Price by fuel type
    st.subheader("Price Distribution by Fuel Type")
    df_fuel = pd.DataFrame(market['fuel_type_stats'])
    # Simulate some spread for boxplot
    expanded = []
    for row in df_fuel.to_dict(orient='records'):
        expanded += [{"fuel_type": row['fuel_type'], "price": np.random.normal(row['avg_price'], 60000)} for _ in range(row['count']//10)]
    df_fuel_exp = pd.DataFrame(expanded)
    box = alt.Chart(df_fuel_exp).mark_boxplot(extent='min-max').encode(
        x=alt.X('fuel_type:N'),
        y=alt.Y('price:Q', title='Price (INR)'),
        color=alt.Color('fuel_type:N', legend=None),
        tooltip=['fuel_type', 'price']
    )
    st.altair_chart(box, use_container_width=True)

    # KPI Cards: Median price, median age, total listings
    median_price = df_brand['avg_price'].median()
    median_age = df_age['vehicle_age'].median()
    total_cars = df_brand['count'].sum()
    st.markdown("## Market KPIs")
    k1, k2, k3 = st.columns(3)
    k1.metric("Median Brand Price", f"₹{median_price:,.0f}")
    k2.metric("Median Vehicle Age", int(median_age))
    k3.metric("Total Listings", int(total_cars))
else:
    st.warning("Market analytics server is unavailable. Showing demo visuals.")

    brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford']
    avg_prices = [600000, 700000, 650000, 750000, 570000]
    ages = list(range(1, 16))
    price_age = [1200000 - (a * 70000) for a in ages]
    bar = alt.Chart(pd.DataFrame({"brand": brands, "avg_price": avg_prices})).mark_bar().encode(
        x="brand", y="avg_price"
    )
    st.altair_chart(bar, use_container_width=True)
    line = alt.Chart(pd.DataFrame({"vehicle_age": ages, "avg_price": price_age})).mark_line(point=True).encode(
        x="vehicle_age", y="avg_price"
    )
    st.altair_chart(line, use_container_width=True)
    st.info("Deploy backend for live insights.")
