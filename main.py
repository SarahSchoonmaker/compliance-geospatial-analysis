#!/usr/bin/env python3
"""
GDPR Compliance Geographic Monitor - Main Script
Optimized modular architecture with high performance
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import warnings
from functools import lru_cache

# Import custom modules - FIXED imports
from src.config import PerformanceConfig, DataConfig, ModelConfig
from src.performance import PerformanceMonitor, SystemInfo, performance_warning
from src.data_generator import SyntheticDataGenerator
from src.model_trainer import ComplianceModelTrainer
from src.dashboard_creator import DashboardCreator
from src.report_generator import ReportGenerator
from src.geographic_setup import GeographicDataSetup

warnings.filterwarnings('ignore')

class GDPRCompliancePredictor:
    """Main orchestrator for GDPR compliance analysis"""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        
        # Initialize components
        self.geo_setup = GeographicDataSetup(self.config)
        self.data_generator = SyntheticDataGenerator(self.config, self.data_config)
        self.model_trainer = ComplianceModelTrainer(self.config, self.model_config)
        self.dashboard_creator = DashboardCreator(self.config, self.data_config)
        self.report_generator = ReportGenerator(self.config)
        
        # Data storage
        self.eu_boundaries = None
        self.data_centers = None
        self.query_data = None
        self.model = None
        
        self.monitor = PerformanceMonitor()
        SystemInfo.log_system_info()
    
    @performance_warning
    def setup_geographic_data(self):
        """Setup EU boundaries and data center locations"""
        self.eu_boundaries, self.data_centers = self.geo_setup.setup_geographic_data()
        return self.eu_boundaries, self.data_centers
    
    @performance_warning
    def generate_synthetic_data(self, n_records: int = None):
        """Generate synthetic query data for analysis"""
        if self.data_centers is None:
            raise ValueError("Geographic data must be setup first. Call setup_geographic_data().")
        
        self.query_data = self.data_generator.generate_synthetic_data(
            n_records or self.data_config.default_records,
            self.data_centers
        )
        return self.query_data
    
    @performance_warning
    def train_compliance_model(self):
        """Train machine learning model for violation prediction"""
        if self.query_data is None:
            raise ValueError("No query data available. Run generate_synthetic_data() first.")
        
        self.model, accuracy = self.model_trainer.train_model(self.query_data)
        return accuracy
    
    @performance_warning
    def create_compliance_dashboard(self):
        """Create interactive Folium dashboard"""
        if self.query_data is None or self.data_centers is None:
            raise ValueError("Missing required data. Ensure data generation and setup are complete.")
        
        dashboard_path = self.dashboard_creator.create_dashboard(
            self.query_data, 
            self.data_centers, 
            self.eu_boundaries
        )
        return dashboard_path
    
    @performance_warning
    def generate_compliance_report(self):
        """Generate comprehensive compliance report"""
        if self.query_data is None:
            raise ValueError("No query data available.")
        
        report_path = self.report_generator.generate_report(
            self.query_data, 
            self.eu_boundaries
        )
        return report_path


def main():
    """Run the complete GDPR compliance analysis"""
    print("🚀 Starting GDPR Compliance Risk Analysis...")
    
    # Initialize with optimized configuration
    config = PerformanceConfig(
        chunk_size=10000,
        n_cores=4,
        use_vectorization=True,
        memory_efficient=True,
        batch_processing=True
    )
    
    predictor = GDPRCompliancePredictor(config)
    
    try:
        # Run analysis pipeline
        print("\n📍 Setting up geographic data...")
        predictor.setup_geographic_data()
        
        print("\n🔄 Generating synthetic query data...")
        predictor.generate_synthetic_data(10000)
        
        print("\n🤖 Training compliance prediction model...")
        accuracy = predictor.train_compliance_model()
        
        print("\n📊 Creating compliance dashboard...")
        predictor.create_compliance_dashboard()
        
        print("\n📋 Generating compliance report...")
        predictor.generate_compliance_report()
        
        print(f"\n✅ Analysis complete! Model accuracy: {accuracy:.1%}")
        print("📊 Open 'compliance-dashboard.html' to view results")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    predictor = main()