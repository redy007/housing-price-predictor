import unittest
import numpy as np
from src.model import train_model
from src.evaluate import evaluate_model

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 8)
        self.y_train = 5 * self.X_train[:, 0] + 3 * self.X_train[:, 1] + np.random.normal(0, 0.1, 100)
        self.X_test = np.random.rand(20, 8)
        self.y_test = 5 * self.X_test[:, 0] + 3 * self.X_test[:, 1] + np.random.normal(0, 0.1, 20)
    
    def test_train_model(self):
        """Test model training."""
        model = train_model(self.X_train, self.y_train)
        
        # Check if model has been trained
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'feature_importances_'))
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        model = train_model(self.X_train, self.y_train)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        # Check if metrics are calculated
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # Check if metrics are reasonable (R² should be > 0 for a decent model)
        self.assertGreater(metrics['r2'], 0)

if __name__ == '__main__':
    unittest.main()
