from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPreprocessing:
    
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        X = self.df.drop("price", axis=1)
        y = self.df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        numeric_features = [
            'bedrooms', 'bathrooms', 'sqft_living',
            'sqft_lot', 'floors', 'sqft_above',
            'sqft_basement', 'house_age'
        ]

        categorical_features = [
            'waterfront', 'view', 'condition',
            'city', 'statezip', 'has_been_renovated'
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        return X_train, X_test, y_train, y_test, preprocessor