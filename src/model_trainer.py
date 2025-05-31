"""
High-performance model training with parallel processing
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.config import PerformanceConfig, ModelConfig
from src.performance import PerformanceMonitor, performance_warning

class ComplianceModelTrainer:
    """Optimized compliance model training"""
    
    def __init__(self, config: PerformanceConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.monitor = PerformanceMonitor()
        self.model = None
    
    @performance_warning
    def train_model(self, query_data: pd.DataFrame):
        """Train compliance model with optimized parameters"""
        self.monitor.start("Model Training")
        
        # Prepare features efficiently
        X = query_data[self.model_config.features].copy()
        y = query_data['violation'].copy()
        
        # Optimized train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
            stratify=y
        )
        
        # Configure model for performance
        self.model = RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            min_samples_split=self.model_config.min_samples_split,
            random_state=self.model_config.random_state,
            n_jobs=self.config.n_cores,  # Use all available cores
            warm_start=True,  # Enable incremental training
            oob_score=True,   # Out-of-bag scoring for efficiency
            bootstrap=True
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate efficiently
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log results
        print(f"📊 Model Accuracy: {accuracy:.3f}")
        print(f"📊 OOB Score: {self.model.oob_score_:.3f}")
        print(f"🎯 Feature Importances:")
        
        # Show top features
        feature_importance = pd.DataFrame({
            'feature': self.model_config.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model efficiently
        if self.config.cache_enabled:
            with open('compliance_model.pkl', 'wb') as f:
                pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.monitor.end()
        return self.model, accuracy
    
    def predict_violations(self, data: pd.DataFrame):
        """Predict violations using trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        features = data[self.model_config.features]
        return self.model.predict_proba(features)[:, 1]  # Return violation probabilities