#!/usr/bin/env python3
"""
GDPR Compliance Geographic Monitor - Main Script
Updated to use modular architecture
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap, FastMarkerCluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import pickle

# Import custom modules
from geographic_setup import PerformanceConfig, DataConfig, ModelConfig, GeoDataConfig, ComplianceConfig
from model_trainer import PerformanceMonitor, SystemInfo, performance_warning
from data_generator import (haversine_distance_vectorized, violation_logic_vectorized, 
                         calculate_time_features, calculate_risk_factors)

warnings.filterwarnings('ignore')

class GDPRCompliancePredictor:
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        
        self.model = None
        self.eu_boundaries = None
        self.data_centers = None
        self.employee_data = None
        self.monitor = PerformanceMonitor()
        
        # Log system information
        SystemInfo.log_system_info()
    
    @lru_cache(maxsize=1000)
    def _get_country_info(self, country: str) -> dict:
        """Cached country information lookup"""
        country_map = ComplianceConfig.get_country_mapping()
        return country_map.get(country, {'is_eu': False, 'strictness': 0.5})
    
    @performance_warning
    def setup_geographic_data(self):
        """Setup EU boundaries and data center locations"""
        self.monitor.start("Geographic Data Setup")
        
        # Get configuration data
        eu_countries = GeoDataConfig.get_eu_countries()
        data_centers = GeoDataConfig.get_data_centers()
        
        # Create EU boundaries GeoDataFrame
        eu_data = []
        for country, info in eu_countries.items():
            eu_data.append({
                'country': country,
                'lat': info['lat'],
                'lon': info['lon'], 
                'gdpr_strictness': info['gdpr_strict']
            })
        
        eu_df = pd.DataFrame(eu_data)
        geometry = gpd.points_from_xy(eu_df.lon, eu_df.lat)
        self.eu_boundaries = gpd.GeoDataFrame(eu_df, geometry=geometry, crs='EPSG:4326')
        
        # Create data centers GeoDataFrame
        dc_data = []
        for dc_id, info in data_centers.items():
            dc_data.append({
                'dc_id': dc_id,
                'lat': info['lat'],
                'lon': info['lon'],
                'country': info['country'],
                'region': info['region'],
                'is_eu': 1 if info['region'] == 'EU' else 0
            })
        
        dc_df = pd.DataFrame(dc_data)
        geometry = gpd.points_from_xy(dc_df.lon, dc_df.lat)
        self.data_centers = gpd.GeoDataFrame(dc_df, geometry=geometry, crs='EPSG:4326')
        
        self.monitor.end()
        return self.eu_boundaries, self.data_centers
    
    # ... (continue with other methods using the modular approach)

def main():
    """Run the complete GDPR compliance analysis"""
    print("🚀 Starting GDPR Compliance Risk Analysis...")
    
    # Initialize predictor with configuration
    config = PerformanceConfig(
        chunk_size=10000,
        n_cores=4,
        use_vectorization=True
    )
    
    predictor = GDPRCompliancePredictor(config)
    
    # Run analysis pipeline
    predictor.setup_geographic_data()
    predictor.generate_synthetic_data(10000)
    predictor.train_compliance_model()
    predictor.create_compliance_dashboard()
    predictor.generate_compliance_report()
    
    print(f"\n✅ Analysis complete! Open 'compliance-dashboard.html' to view results")
    return predictor

if __name__ == "__main__":
    predictor = main()