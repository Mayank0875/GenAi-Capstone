import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# Import your custom classes
from components.data_cleaning import DataCleaning
from components.data_preprocessing import DataPreprocessing


def main():

    print("Loading dataset...")
    df = pd.read_csv("data/raw_data.csv")

    print("Cleaning data...")
    cleaner = DataCleaning(df)
    df_clean = cleaner.clean_data()

    print("Preprocessing data...")
    preprocessor_obj = DataPreprocessing(df_clean)
    X_train, X_test, y_train, y_test, preprocessor = preprocessor_obj.preprocess()

    print("Building model pipeline...")

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'   # Fast training
        ))
    ])

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")

    print("Saving model...")
    joblib.dump(model_pipeline, "house_price_model.pkl")

    print("Model saved as house_price_model.pkl")


if __name__ == "__main__":
    main()