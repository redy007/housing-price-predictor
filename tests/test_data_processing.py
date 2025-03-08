import unittest
import pandas as pd
import numpy as np
import os
from src.data_processing import load_data, preprocess_data

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'longitude': [-122.23, -122.22],
            'latitude': [37.88, 37.86],
            'housing_median_age': [41.0, 21.0],
            'total_rooms': [880.0, 7099.0],
            'total_bedrooms': [129.0, 1106.0],
            'population': [322.0, 2401.0],
            'households': [126.0, 1138.0],
            'median_income': [8.3252, 8.3014],
            'median_house_value': [452600.0, 358500.0]
        })
        
        # Save the sample data for testing load_data
        os.makedirs('data', exist_ok=True)
        self.sample_data.to_csv('data/test_housing.csv', index=False)
    
    def test_load_data(self):
        """Test if data is loaded correctly."""
        df = load_data('data/test_housing.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), list(self.sample_data.columns))
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X_train, X_test, y_train, y_test, scaler = preprocess_data(self.sample_data)
        
        # Check shapes
        self.assertEqual(X_train.shape[1], 8)  # 8 features
        self.assertEqual(len(y_train) + len(y_test), 2)  # 2 samples total
        
        # Check if data is scaled (mean close to 0)
        self.assertTrue(np.abs(X_train.mean(axis=0)).max() < 1e-10)
        
    def tearDown(self):
        if os.path.exists('data/test_housing.csv'):
            os.remove('data/test_housing.csv')

if __name__ == '__main__':
    unittest.main()
