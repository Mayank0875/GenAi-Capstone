import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from src.data_preprocessing import DataPreprocessing

# Load cleaned train data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Recreate preprocessing
df_dummy = pd.concat([X_train, y_train], axis=1)
preprocessor_obj = DataPreprocessing(df_dummy)
_, _, _, _, preprocessor = preprocessor_obj.preprocess()

# Create pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Grid parameters
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Evaluate
y_pred = grid.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(grid.best_estimator_, "data/best_model.pkl")

print("Model saved successfully")