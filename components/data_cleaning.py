import pandas as pd
import numpy as np

class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_outliers(self, column):
        q25, q75 = np.percentile(self.df[column], 25), np.percentile(self.df[column], 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        upper = q75 + cut_off
        
        self.df = self.df[(self.df[column] < upper) & (self.df[column] > 1)]
        print(f"Outliers removed from {column}")
        return self.df

    def feature_engineering(self):
        self.df['year_sold'] = pd.to_datetime(self.df['date']).dt.year
        self.df['house_age'] = self.df['year_sold'] - self.df['yr_built']
        self.df['has_been_renovated'] = self.df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

        self.df.drop(['date', 'yr_renovated', 'yr_built', 
                      'street', 'country'], axis=1, inplace=True)

        print("Feature engineering completed")
        return self.df

    def clean_data(self):
        self.remove_outliers('price')
        self.remove_outliers('sqft_lot')
        self.feature_engineering()
        return self.df