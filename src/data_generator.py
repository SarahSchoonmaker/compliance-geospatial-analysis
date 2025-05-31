"""
High-performance synthetic data generation for Amazon-EU merchant GDPR scenario
CORRECTED: Models US Amazon employees accessing EU merchant data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

from src.config import PerformanceConfig, DataConfig, GeoDataConfig, ComplianceConfig
from src.performance import PerformanceMonitor, performance_warning
from src.optimization import (
    haversine_distance_vectorized, 
    violation_logic_vectorized, 
    calculate_time_features, 
    calculate_risk_factors,
    batch_process_violations
)

class SyntheticDataGenerator:
    """Optimized synthetic data generation for Amazon-EU merchant scenario"""
    
    def __init__(self, config: PerformanceConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self.monitor = PerformanceMonitor()
    
    @lru_cache(maxsize=1000)
    def _get_country_info(self, country: str) -> dict:
        """Cached country information lookup"""
        country_map = ComplianceConfig.get_country_mapping()
        return country_map.get(country, {'is_eu': False, 'strictness': 0.5})
    
    @performance_warning
    def generate_synthetic_data(self, n_records: int, data_centers: pd.DataFrame):
        """Generate synthetic query data using vectorized operations"""
        self.monitor.start(f"Synthetic Data Generation ({n_records:,} records)")
        
        np.random.seed(self.data_config.random_seed)
        
        if self.config.batch_processing and n_records > self.config.chunk_size:
            # Process in chunks for memory efficiency
            query_data = self._generate_in_batches(n_records, data_centers)
        else:
            # Generate all at once
            query_data = self._generate_batch(n_records, data_centers)
        
        self.monitor.end()
        return query_data
    
    def _generate_in_batches(self, n_records: int, data_centers: pd.DataFrame):
        """Generate data in memory-efficient batches"""
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, n_records, chunk_size):
            current_chunk_size = min(chunk_size, n_records - i)
            chunk = self._generate_batch(current_chunk_size, data_centers, start_id=i)
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _generate_batch(self, n_records: int, data_centers: pd.DataFrame, start_id: int = 0):
        """Generate Amazon-EU merchant scenario data"""
        employee_locations = GeoDataConfig.get_employee_locations()
        
        # FIXED: Ensure array sizes match
        n_locations = len(employee_locations)
        location_weights = self.data_config.location_weights
        
        # Ensure weights array matches locations array size
        if len(location_weights) != n_locations:
            # Auto-generate weights based on realistic Amazon employee distribution
            location_weights = [loc.get('weight', 1.0/n_locations) for loc in employee_locations]
            print(f"⚠️ Using location-specific weights for {n_locations} Amazon offices")
        
        # Normalize weights to sum to 1
        location_weights = np.array(location_weights)
        location_weights = location_weights / location_weights.sum()
        
        # Generate Amazon employee locations
        employee_choices = np.random.choice(
            n_locations,
            n_records, 
            p=location_weights
        )
        
        # Generate data center assignments (where data is accessed from)
        dc_choices = np.random.choice(
            data_centers['dc_id'].values, 
            n_records
        )
        
        # Generate timestamps with realistic patterns
        random_days = np.random.randint(0, 365, n_records)
        base_timestamp = datetime.now()
        timestamps = [base_timestamp - timedelta(days=int(days)) for days in random_days]
        
        # Create query data
        query_ids = [f'AMZ{i+start_id+1:06d}' for i in range(n_records)]  # Amazon query IDs
        emp_locs = [employee_locations[i] for i in employee_choices]
        
        # Build data efficiently
        query_data = pd.DataFrame({
            'query_id': query_ids,
            'employee_city': [loc['city'] for loc in emp_locs],
            'employee_country': [loc['country'] for loc in emp_locs],
            'employee_lat': [loc['lat'] for loc in emp_locs],
            'employee_lon': [loc['lon'] for loc in emp_locs],
            'data_center': dc_choices,
            'timestamp': timestamps
        })
        
        # Add data center info
        dc_info = data_centers.set_index('dc_id')[['country', 'lat', 'lon', 'region']].add_prefix('dc_')
        query_data = query_data.join(dc_info, on='data_center')
        
        # Add time features
        query_data['hour'] = [ts.hour for ts in timestamps]
        query_data['day_of_week'] = [ts.weekday() for ts in timestamps]
        
        # CORRECTED: Generate merchant data with realistic EU merchant distribution
        # 40% of queries access merchant data, 60% access EU merchants (of merchant queries)
        is_merchant_data = np.random.choice([0, 1], n_records, p=[0.6, 0.4])
        query_data['is_merchant_data'] = is_merchant_data
        
        # Generate merchant countries - focus on EU merchants for GDPR scenario
        merchant_countries = self._generate_merchant_countries(n_records, is_merchant_data)
        query_data['merchant_country'] = merchant_countries
        
        # Add query types
        query_data['query_type'] = np.random.choice(
            ['SELECT', 'UPDATE', 'INSERT', 'DELETE'], n_records,
            p=[0.7, 0.15, 0.10, 0.05]  # Realistic query distribution
        )
        
        # Calculate all derived features efficiently
        self._calculate_features_vectorized(query_data)
        
        return query_data
    
    def _generate_merchant_countries(self, n_records: int, is_merchant_data: np.ndarray):
        """Generate realistic merchant country distribution for Amazon"""
        merchant_countries = [''] * n_records
        
        # EU countries where Amazon has significant merchant presence
        eu_merchant_countries = ['Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
                                'Poland', 'Sweden', 'Belgium', 'Austria', 'Ireland']
        us_merchant_countries = ['USA']
        other_countries = ['Japan', 'India', 'Brazil', 'Canada', 'Australia']
        
        for i in range(n_records):
            if is_merchant_data[i]:
                # 65% EU merchants, 25% US merchants, 10% other
                rand_val = np.random.random()
                if rand_val < 0.65:
                    merchant_countries[i] = np.random.choice(eu_merchant_countries)
                elif rand_val < 0.90:
                    merchant_countries[i] = np.random.choice(us_merchant_countries)
                else:
                    merchant_countries[i] = np.random.choice(other_countries)
            else:
                merchant_countries[i] = 'N/A'  # Not merchant data
        
        return merchant_countries
    
    def _calculate_features_vectorized(self, query_data: pd.DataFrame):
        """Calculate all derived features for Amazon-EU merchant scenario"""
        
        # Distance calculation (Numba optimized)
        distances = haversine_distance_vectorized(
            query_data['employee_lat'].values,
            query_data['employee_lon'].values,
            query_data['dc_lat'].values,
            query_data['dc_lon'].values
        )
        query_data['distance_km'] = distances
        
        # Time-based features (Numba optimized)
        off_hours, weekend = calculate_time_features(
            query_data['hour'].values,
            query_data['day_of_week'].values
        )
        query_data['off_hours'] = off_hours
        query_data['weekend'] = weekend
        
        # Geographic risk factors (Numba optimized)
        suspicious_distance = calculate_risk_factors(distances)
        query_data['suspicious_distance'] = suspicious_distance
        
        # CORRECTED: Employee country classification for Amazon scenario
        query_data['is_us_employee'] = (query_data['employee_country'] == 'USA').astype(int)
        query_data['is_eu_employee'] = query_data['employee_country'].apply(
            lambda x: 1 if self._get_country_info(x)['is_eu'] else 0
        )
        
        # CORRECTED: Merchant data classification
        eu_countries = ['Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
                       'Poland', 'Sweden', 'Belgium', 'Austria', 'Ireland']
        query_data['is_eu_merchant_data'] = (
            (query_data['is_merchant_data'] == 1) & 
            (query_data['merchant_country'].isin(eu_countries))
        ).astype(int)
        
        # Data center location classification
        query_data['is_non_eu_data_center'] = (query_data['dc_region'] != 'EU').astype(int)
        
        # GDPR strictness scores for merchant countries
        merchant_strictness = []
        for country in query_data['merchant_country']:
            if country in eu_countries:
                country_info = self._get_country_info(country)
                merchant_strictness.append(country_info.get('strictness', 1.0))
            else:
                merchant_strictness.append(0.0)  # Non-EU merchants not subject to GDPR
        query_data['merchant_gdpr_strictness'] = merchant_strictness
        
        # CORRECTED: Generate violations using the corrected logic
        # US Amazon employees accessing EU merchant data = GDPR violation
        if self.config.use_vectorization:
            random_vals = np.random.random(len(query_data))
            violations = violation_logic_vectorized(
                query_data['is_us_employee'].values,           # US employees
                query_data['is_eu_merchant_data'].values,      # EU merchant data
                query_data['is_merchant_data'].values,         # Any merchant data
                query_data['off_hours'].values,
                query_data['suspicious_distance'].values,
                random_vals,
                np.array(query_data['merchant_gdpr_strictness'].values, dtype=np.float32)
            )
        else:
            # Fallback to batch processing
            violations = batch_process_violations(
                query_data['is_us_employee'].values,
                query_data['is_eu_merchant_data'].values,
                query_data['is_merchant_data'].values,
                query_data['off_hours'].values,
                query_data['suspicious_distance'].values,
                np.random.random(len(query_data))
            )
        
        query_data['violation'] = violations
        
        # Add violation reason for analysis
        violation_reasons = []
        for idx, row in query_data.iterrows():
            if row['violation'] == 1:
                if row['is_us_employee'] and row['is_eu_merchant_data']:
                    violation_reasons.append('US_employee_EU_merchant_data')
                elif row['off_hours'] and row['is_merchant_data']:
                    violation_reasons.append('off_hours_merchant_access')
                elif row['suspicious_distance'] and row['is_merchant_data']:
                    violation_reasons.append('suspicious_distance_merchant_access')
                else:
                    violation_reasons.append('other_risk_factors')
            else:
                violation_reasons.append('no_violation')
        
        query_data['violation_reason'] = violation_reasons