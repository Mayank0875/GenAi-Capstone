import streamlit as st
import os
from predict import predict_house_price

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction App")
st.write("Fill all details below:")


bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=1800)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=0, value=5000)
floors = st.number_input("Floors", min_value=0, max_value=5, value=1)
sqft_above = st.number_input("Sqft Above Ground", min_value=0.0, value=1800.0)
sqft_basement = st.number_input("Sqft Basement", min_value=0, value=0)
house_age = st.number_input("House Age (years)", min_value=0, value=20)

waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View (0–4)", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition (1–5)", [1, 2, 3, 4, 5])

city = st.text_input("City", value="Seattle")
statezip = st.text_input("State + Zip (e.g., WA 98178)", value="WA 98178")

has_been_renovated = st.selectbox("Has Been Renovated", [0, 1])

# Derived Features
total_sqft = sqft_living + sqft_basement
bath_per_bed = bathrooms / (bedrooms + 1)


if st.button("Predict Price"):
    input_data = {
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": int(floors),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "house_age": int(house_age),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "city": city,
        "statezip": statezip,
        "has_been_renovated": int(has_been_renovated),
        "total_sqft": int(total_sqft),
        "bath_per_bed": float(bath_per_bed)
    }

    price = predict_house_price(input_data)

    st.success(f"Predicted House Price: ${price:,.2f}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")