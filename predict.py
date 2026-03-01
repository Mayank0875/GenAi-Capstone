import joblib
import pandas as pd

# Load model once (global)
model = joblib.load("house_price_model.pkl")


def predict_house_price(input_data: dict):
    """
    Predict house price.

    Parameters:
    -----------
    input_data : dict
        Dictionary containing house features.

    Returns:
    --------
    float
        Predicted house price
    """

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df)

    return float(prediction[0])


# Example usage
if __name__ == "__main__":

    sample_input = {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "house_age": 20,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "city": "Seattle",
        "statezip": "WA 98178",
        "has_been_renovated": 0,
        "total_sqft": 1800,
        "bath_per_bed": 0.66
    }

    price = predict_house_price(sample_input)

    print(f"Predicted Price: ${price:,.2f}")