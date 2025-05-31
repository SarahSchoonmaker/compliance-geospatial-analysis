#!/usr/bin/env python3
"""
High-Performance GDPR Data Access Compliance Risk Predictor
Optimized for scale and production use
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap, FastMarkerCluster
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime, timedelta
import warnings
import multiprocessing as mp
from functools import lru_cache
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numba
from numba import jit, njit
import psutil

warnings.filterwarnings('ignore')

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    chunk_size: int = 10000
    n_cores: int = min(mp.cpu_count(), 8)  # Limit cores to prevent memory issues
    use_vectorization: bool = True
    cache_enabled: bool = True
    batch_processing: bool = True
    memory_efficient: bool = True

class PerformanceMonitor:
    """Monitor memory usage and execution time"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        logger.info(f"🚀 Starting {operation_name}")
    
    def end(self):
        end_time = datetime.now()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        duration = (end_time - self.start_time).total_seconds()
        memory_diff = end_memory - self.start_memory
        
        logger.info(f"✅ {self.operation_name} completed in {duration:.2f}s")
        logger.info(f"📊 Memory usage: {memory_diff:+.1f}MB (Total: {end_memory:.1f}MB)")

# Vectorized distance calculation using Numba
@njit
def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation using Numba JIT"""
    R = 6371  # Earth's radius in km
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

@njit
def violation_logic_vectorized(is_eu_employee, is_non_eu_data, is_merchant_data, 
                              off_hours, suspicious_distance, random_vals):
    """Vectorized GDPR violation logic using Numba"""
    violations = np.zeros(len(is_eu_employee), dtype=np.int32)
    
    for i in range(len(is_eu_employee)):
        if is_eu_employee[i] and is_non_eu_data[i] and is_merchant_data[i]:
            violations[i] = 1
        elif off_hours[i] and is_merchant_data[i] and random_vals[i] < 0.3:
            violations[i] = 1
        elif suspicious_distance[i] and is_merchant_data[i] and random_vals[i] < 0.2:
            violations[i] = 1
    
    return violations

class OptimizedGDPRPredictor:
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.model = None
        self.eu_boundaries = None
        self.data_centers = None
        self.employee_data = None
        self.monitor = PerformanceMonitor()
        
        # Pre-computed lookup tables for performance
        self._distance_cache = {}
        self._employee_lookup = {}
        
        logger.info(f"🔧 Initialized with {self.config.n_cores} cores, chunk size {self.config.chunk_size}")
    
    @lru_cache(maxsize=1000)
    def _get_country_info(self, country: str) -> Dict:
        """Cached country information lookup"""
        country_map = {
            'Ireland': {'is_eu': True, 'strictness': 1.0},
            'Germany': {'is_eu': True, 'strictness': 1.0},
            'UK': {'is_eu': True, 'strictness': 0.9},  # Post-Brexit but still strict
            'Netherlands': {'is_eu': True, 'strictness': 1.0},
            'France': {'is_eu': True, 'strictness': 1.0},
            'USA': {'is_eu': False, 'strictness': 0.3},
            'India': {'is_eu': False, 'strictness': 0.2},
        }
        return country_map.get(country, {'is_eu': False, 'strictness': 0.5})
    
    def setup_geographic_data(self):
        """Optimized geographic data setup with caching"""
        self.monitor.start("Geographic Data Setup")
        
        # Pre-defined data for performance (avoid dynamic lookups)
        eu_countries_data = np.array([
            [51.1657, 10.4515, 1.0],  # Germany
            [46.2276, 2.2137, 1.0],   # France
            [53.1424, -7.6921, 1.0],  # Ireland
            [52.1326, 5.2913, 1.0],   # Netherlands
            [60.1282, 18.6435, 1.0],  # Sweden
            [51.9194, 19.1451, 0.8],  # Poland
            [40.4637, -3.7492, 0.9],  # Spain
            [41.8719, 12.5674, 0.7],  # Italy
        ])
        
        countries = ['Germany', 'France', 'Ireland', 'Netherlands', 'Sweden', 'Poland', 'Spain', 'Italy']
        
        # Create optimized GeoDataFrame
        eu_df = pd.DataFrame({
            'country': countries,
            'lat': eu_countries_data[:, 0],
            'lon': eu_countries_data[:, 1],
            'gdpr_strictness': eu_countries_data[:, 2]
        })
        
        # Vectorized geometry creation
        geometry = gpd.points_from_xy(eu_df.lon, eu_df.lat)
        self.eu_boundaries = gpd.GeoDataFrame(eu_df, geometry=geometry, crs='EPSG:4326')
        
        # Data centers with pre-computed attributes
        dc_data = np.array([
            [53.3498, -6.2603, 1],    # eu-west-1 (Ireland)
            [50.1109, 8.6821, 1],     # eu-central-1 (Germany)
            [51.5074, -0.1278, 1],    # eu-west-2 (UK)
            [39.0458, -77.5081, 0],   # us-east-1 (USA)
            [45.5152, -122.6784, 0],  # us-west-2 (USA)
            [1.3521, 103.8198, 0],    # ap-southeast-1 (Singapore)
        ])
        
        dc_ids = ['eu-west-1', 'eu-central-1', 'eu-west-2', 'us-east-1', 'us-west-2', 'ap-southeast-1']
        dc_countries = ['Ireland', 'Germany', 'UK', 'USA', 'USA', 'Singapore']
        dc_regions = ['EU', 'EU', 'EU', 'US', 'US', 'APAC']
        
        dc_df = pd.DataFrame({
            'dc_id': dc_ids,
            'lat': dc_data[:, 0],
            'lon': dc_data[:, 1],
            'is_eu': dc_data[:, 2],
            'country': dc_countries,
            'region': dc_regions
        })
        
        geometry = gpd.points_from_xy(dc_df.lon, dc_df.lat)
        self.data_centers = gpd.GeoDataFrame(dc_df, geometry=geometry, crs='EPSG:4326')
        
        self.monitor.end()
        return self.eu_boundaries, self.data_centers
    
    def generate_synthetic_data_optimized(self, n_records: int = 100000):
        """Highly optimized synthetic data generation using vectorization"""
        self.monitor.start(f"Synthetic Data Generation ({n_records:,} records)")
        
        # Pre-allocate arrays for performance
        np.random.seed(42)
        
        # Employee locations with weights (optimized lookup)
        locations = np.array([
            [53.3498, -6.2603, 0, 0.15],  # Dublin, Ireland
            [52.5200, 13.4050, 1, 0.10],  # Berlin, Germany
            [51.5074, -0.1278, 2, 0.20],  # London, UK
            [47.6062, -122.3321, 3, 0.25], # Seattle, USA
            [39.0458, -77.5081, 3, 0.20],  # Virginia, USA
            [19.0760, 72.8777, 4, 0.10],   # Mumbai, India
        ])
        
        location_names = ['Dublin', 'Berlin', 'London', 'Seattle', 'Virginia', 'Mumbai']
        country_names = ['Ireland', 'Germany', 'UK', 'USA', 'USA', 'India']
        
        # Vectorized random sampling
        location_indices = np.random.choice(6, size=n_records, p=locations[:, 3])
        
        # Vectorized coordinate generation with noise
        emp_lats = locations[location_indices, 0] + np.random.normal(0, 0.05, n_records)
        emp_lons = locations[location_indices, 1] + np.random.normal(0, 0.05, n_records)
        
        # Vectorized data center selection
        dc_indices = np.random.randint(0, len(self.data_centers), n_records)
        dc_data = self.data_centers.iloc[dc_indices]
        
        # Vectorized time generation
        days_back = np.random.randint(0, 90, n_records)
        base_time = datetime.now()
        query_times = [base_time - timedelta(days=int(d)) for d in days_back]
        hours = np.array([qt.hour for qt in query_times])
        day_of_weeks = np.array([qt.weekday() for qt in query_times])
        
        # Vectorized distance calculation using optimized haversine
        distances = haversine_distance_vectorized(
            emp_lats, emp_lons,
            dc_data.lat.values, dc_data.lon.values
        )
        
        # Vectorized boolean operations
        employee_countries = np.array([country_names[i] for i in location_indices])
        is_eu_employee = np.isin(employee_countries, ['Ireland', 'Germany', 'UK', 'Netherlands', 'France'])
        is_non_eu_data = dc_data.is_eu.values == 0
        is_merchant_data = np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])
        off_hours = (hours < 6) | (hours > 22)
        weekend = day_of_weeks >= 5
        suspicious_distance = distances > 5000
        
        # Vectorized violation logic using Numba
        random_vals = np.random.random(n_records)
        violations = violation_logic_vectorized(
            is_eu_employee, is_non_eu_data, is_merchant_data.astype(bool),
            off_hours, suspicious_distance, random_vals
        )
        
        # Efficiently create DataFrame
        self.employee_data = pd.DataFrame({
            'employee_id': [f'EMP_{i:06d}' for i in range(n_records)],
            'employee_lat': emp_lats,
            'employee_lon': emp_lons,
            'employee_country': employee_countries,
            'employee_city': [location_names[i] for i in location_indices],
            'data_center_id': dc_data.dc_id.values,
            'dc_lat': dc_data.lat.values,
            'dc_lon': dc_data.lon.values,
            'dc_region': dc_data.region.values,
            'dc_country': dc_data.country.values,
            'hour': hours,
            'day_of_week': day_of_weeks,
            'is_merchant_data': is_merchant_data,
            'distance_km': distances,
            'is_eu_employee': is_eu_employee.astype(int),
            'is_non_eu_data': is_non_eu_data.astype(int),
            'off_hours': off_hours.astype(int),
            'weekend': weekend.astype(int),
            'gdpr_violation': violations
        })
        
        self.monitor.end()
        logger.info(f"📊 Violation rate: {self.employee_data.gdpr_violation.mean():.1%}")
        
        return self.employee_data
    
    def train_compliance_model_optimized(self):
        """Optimized model training with parallel processing"""
        self.monitor.start("Model Training")
        
        # Feature selection optimized for performance
        features = [
            'distance_km', 'is_merchant_data', 'is_eu_employee', 
            'is_non_eu_data', 'off_hours', 'weekend', 'hour', 'day_of_week'
        ]
        
        X = self.employee_data[features].values  # Use numpy arrays for speed
        y = self.employee_data['gdpr_violation'].values
        
        # Optimized train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Optimized Random Forest with parallel processing
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=self.config.n_cores,  # Use all available cores
            max_depth=10,  # Limit depth for speed
            min_samples_split=100,  # Speed up training
            bootstrap=True
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Batch prediction for efficiency
        batch_size = 10000
        n_batches = len(X) // batch_size + 1
        
        violation_probs = np.zeros(len(X))
        predictions = np.zeros(len(X))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            
            if start_idx < len(X):
                batch_X = X[start_idx:end_idx]
                violation_probs[start_idx:end_idx] = self.model.predict_proba(batch_X)[:, 1]
                predictions[start_idx:end_idx] = self.model.predict(batch_X)
        
        self.employee_data['violation_probability'] = violation_probs
        self.employee_data['predicted_violation'] = predictions
        
        self.monitor.end()
        logger.info(f"✅ Model trained with {accuracy:.1%} accuracy")
        
        return accuracy
    
    def create_optimized_dashboard(self, max_points: int = 1000):
        """Create optimized dashboard with clustering for large datasets"""
        self.monitor.start("Dashboard Creation")
        
        # Center map on Europe
        m = folium.Map(location=[52.5, 10.0], zoom_start=4)
        
        # Add EU countries (optimized)
        for _, country in self.eu_boundaries.iterrows():
            folium.CircleMarker(
                location=[country.lat, country.lon],
                radius=6,
                popup=f"<b>{country.country}</b><br>GDPR: {country.gdpr_strictness}",
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.6
            ).add_to(m)
        
        # Add data centers
        for _, dc in self.data_centers.iterrows():
            color = 'red' if dc.is_eu == 0 else 'green'
            folium.Marker(
                location=[dc.lat, dc.lon],
                popup=f"<b>{dc.dc_id}</b><br>{dc.region}",
                icon=folium.Icon(color=color, icon='server', prefix='fa')
            ).add_to(m)
        
        # Optimized high-risk points with clustering
        high_risk = self.employee_data[self.employee_data.violation_probability > 0.7]
        
        if len(high_risk) > max_points:
            # Sample for performance
            high_risk = high_risk.sample(max_points)
            logger.info(f"📊 Sampling {max_points} points from {len(high_risk)} high-risk queries")
        
        # Use FastMarkerCluster for performance
        risk_points = [[row.employee_lat, row.employee_lon] for _, row in high_risk.iterrows()]
        
        if risk_points:
            FastMarkerCluster(risk_points, name='High Risk Locations').add_to(m)
            
            # Optimized heatmap
            heatmap_data = [[row.employee_lat, row.employee_lon, row.violation_probability] 
                           for _, row in high_risk.iterrows()]
            
            HeatMap(
                heatmap_data, 
                name='Violation Risk Heatmap',
                radius=15,
                blur=10,
                max_zoom=1
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save('gdpr_compliance_dashboard.html')
        
        self.monitor.end()
        logger.info("✅ Dashboard saved as 'gdpr_compliance_dashboard.html'")
        
        return m
    
    def generate_optimized_report(self):
        """Generate optimized analytics report using vectorized operations"""
        self.monitor.start("Report Generation")
        
        # Vectorized calculations
        total_queries = len(self.employee_data)
        violation_mask = self.employee_data['gdpr_violation'] == 1
        high_risk_mask = self.employee_data['violation_probability'] > 0.7
        
        actual_violations = violation_mask.sum()
        predicted_high_risk = high_risk_mask.sum()
        
        print("\n" + "="*60)
        print("📊 OPTIMIZED GDPR COMPLIANCE ANALYSIS REPORT")
        print("="*60)
        print(f"Total Queries Analyzed: {total_queries:,}")
        print(f"Actual Violations: {actual_violations:,} ({actual_violations/total_queries:.1%})")
        print(f"High-Risk Queries: {predicted_high_risk:,} ({predicted_high_risk/total_queries:.1%})")
        
        # Optimized groupby operations
        print(f"\n🌍 Geographic Distribution:")
        country_stats = self.employee_data.groupby('employee_country')['gdpr_violation'].agg(['count', 'sum', 'mean']).round(3)
        print(country_stats)
        
        print(f"\n💾 Data Center Risk Analysis:")
        dc_stats = self.employee_data.groupby(['data_center_id', 'dc_region']).agg({
            'violation_probability': 'mean',
            'gdpr_violation': 'sum'
        }).round(3)
        print(dc_stats)
        
        self.monitor.end()
    
    def save_model(self, filepath: str = 'gdpr_model.pkl'):
        """Save trained model for reuse"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'gdpr_model.pkl'):
        """Load pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"📂 Model loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Model file {filepath} not found")

def main_optimized():
    """Run optimized GDPR compliance analysis"""
    
    print("🚀 Starting OPTIMIZED GDPR Compliance Analysis...")
    
    # Performance configuration
    config = PerformanceConfig(
        chunk_size=50000,
        n_cores=min(mp.cpu_count(), 6),
        use_vectorization=True,
        cache_enabled=True
    )
    
    # Initialize optimized predictor
    predictor = OptimizedGDPRPredictor(config)
    
    # Setup data
    predictor.setup_geographic_data()
    
    # Generate larger dataset to demonstrate performance
    predictor.generate_synthetic_data_optimized(100000)
    
    # Train model
    predictor.train_compliance_model_optimized()
    
    # Save model for reuse
    predictor.save_model()
    
    # Create dashboard
    predictor.create_optimized_dashboard()
    
    # Generate report
    predictor.generate_optimized_report()
    
    print(f"\n✅ Optimized analysis complete! Check 'gdpr_compliance_dashboard.html'")
    
    return predictor

if __name__ == "__main__":
    predictor = main_optimized()