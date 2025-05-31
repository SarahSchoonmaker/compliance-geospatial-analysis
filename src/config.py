"""
High-Performance Configuration Module for GDPR Compliance Analysis
Optimized for Amazon-EU merchant scenario with memory efficiency and scalability
"""

import multiprocessing as mp
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class OptimizationLevel(Enum):
    """Optimization levels for different use cases"""
    DEVELOPMENT = "development"      # Fast iteration, moderate performance
    PRODUCTION = "production"        # Maximum performance
    MEMORY_CONSTRAINED = "memory"    # Optimized for low memory environments
    HIGH_THROUGHPUT = "throughput"   # Optimized for processing large datasets

@dataclass
class PerformanceConfig:
    """Advanced performance configuration with auto-optimization"""
    
    # Core processing parameters
    chunk_size: int = field(default_factory=lambda: PerformanceConfig._auto_chunk_size())
    n_cores: int = field(default_factory=lambda: PerformanceConfig._auto_cores())
    memory_limit_gb: float = field(default_factory=lambda: PerformanceConfig._auto_memory_limit())
    
    # Optimization toggles
    use_vectorization: bool = True
    use_numba_jit: bool = True
    cache_enabled: bool = True
    batch_processing: bool = True
    memory_efficient: bool = True
    parallel_processing: bool = True
    
    # Advanced performance features
    optimization_level: OptimizationLevel = OptimizationLevel.PRODUCTION
    enable_profiling: bool = False
    gc_frequency: int = 1000  # Garbage collection frequency
    
    # Memory management
    max_memory_usage_percent: float = 80.0
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 70.0
    
    # Caching configuration
    lru_cache_size: int = 10000
    enable_disk_cache: bool = False
    cache_directory: str = ".cache"
    
    @staticmethod
    def _auto_chunk_size() -> int:
        """Automatically determine optimal chunk size based on available memory"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb >= 16:
            return 50000  # Large chunks for high-memory systems
        elif available_memory_gb >= 8:
            return 25000  # Medium chunks
        elif available_memory_gb >= 4:
            return 10000  # Smaller chunks for limited memory
        else:
            return 5000   # Very small chunks for constrained systems
    
    @staticmethod
    def _auto_cores() -> int:
        """Automatically determine optimal number of cores"""
        total_cores = mp.cpu_count()
        logical_cores = psutil.cpu_count(logical=True)
        
        # Use physical cores for CPU-intensive tasks, leave some for system
        if total_cores >= 8:
            return min(total_cores - 2, 12)  # Leave 2 cores for system, cap at 12
        elif total_cores >= 4:
            return total_cores - 1
        else:
            return max(1, total_cores - 1)
    
    @staticmethod
    def _auto_memory_limit() -> float:
        """Automatically determine memory limit based on available memory"""
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        return total_memory_gb * 0.7  # Use 70% of total memory
    
    def get_optimized_config(self, dataset_size: int) -> 'PerformanceConfig':
        """Get optimized configuration based on dataset size"""
        config = PerformanceConfig()
        
        # Adjust settings based on dataset size
        if dataset_size > 1_000_000:  # Large dataset
            config.optimization_level = OptimizationLevel.HIGH_THROUGHPUT
            config.chunk_size = min(100000, self.chunk_size * 2)
            config.batch_processing = True
            config.parallel_processing = True
            config.gc_frequency = 500
        elif dataset_size > 100_000:  # Medium dataset
            config.optimization_level = OptimizationLevel.PRODUCTION
            config.chunk_size = self.chunk_size
        else:  # Small dataset
            config.optimization_level = OptimizationLevel.DEVELOPMENT
            config.chunk_size = min(10000, dataset_size)
            config.parallel_processing = dataset_size > 10000
        
        return config

@dataclass
class DataConfig:
    """Optimized data generation and processing configuration"""
    
    # Data generation parameters
    default_records: int = 100000
    max_dashboard_points: int = 2000  # Increased for better visualization
    random_seed: int = 42
    
    # Sampling configuration
    stratified_sampling: bool = True
    min_violation_samples: int = 100
    sample_balance_ratio: float = 0.3  # 30% violations in samples
    
    # Data types optimization
    use_categorical_dtypes: bool = True
    compress_strings: bool = True
    optimize_memory_usage: bool = True
    
    # Geographic data configuration
    coordinate_precision: int = 4  # Decimal places for lat/lon
    distance_precision: int = 1    # Decimal places for distances
    
    # Amazon employee location weights (realistic distribution)
    location_weights: List[float] = field(default_factory=lambda: [0.30, 0.25, 0.15, 0.12, 0.08, 0.06, 0.04])
    
    # Data validation
    enable_data_validation: bool = True
    max_distance_km: float = 20000.0  # Maximum realistic distance
    min_distance_km: float = 0.1      # Minimum distance threshold
    
    def get_optimal_dtypes(self) -> Dict[str, str]:
        """Get optimized data types for memory efficiency"""
        return {
            'query_id': 'string',
            'employee_city': 'category',
            'employee_country': 'category',
            'data_center': 'category',
            'dc_country': 'category',
            'dc_region': 'category',
            'query_type': 'category',
            'merchant_country': 'category',
            'violation_reason': 'category',
            'hour': 'int8',
            'day_of_week': 'int8',
            'is_merchant_data': 'int8',
            'is_us_employee': 'int8',
            'is_eu_employee': 'int8',
            'is_eu_merchant_data': 'int8',
            'is_non_eu_data_center': 'int8',
            'off_hours': 'int8',
            'weekend': 'int8',
            'suspicious_distance': 'int8',
            'violation': 'int8',
            'distance_km': 'float32',
            'employee_lat': 'float32',
            'employee_lon': 'float32',
            'dc_lat': 'float32',
            'dc_lon': 'float32',
            'merchant_gdpr_strictness': 'float32'
        }

@dataclass
class ModelConfig:
    """Optimized machine learning model configuration"""
    
    # Model parameters
    n_estimators: int = 200  # Increased for better performance
    max_depth: int = 15      # Increased depth
    min_samples_split: int = 50  # More aggressive splitting
    min_samples_leaf: int = 20
    max_features: str = 'sqrt'
    
    # Training configuration
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Performance optimization
    enable_early_stopping: bool = True
    warm_start: bool = True
    bootstrap: bool = True
    oob_score: bool = True
    
    # Feature engineering for Amazon-EU merchant scenario
    features: List[str] = field(default_factory=lambda: [
        'distance_km', 'is_merchant_data', 'is_us_employee', 
        'is_eu_merchant_data', 'off_hours', 'weekend', 'hour', 
        'day_of_week', 'suspicious_distance', 'merchant_gdpr_strictness'
    ])
    
    # Advanced features
    interaction_features: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('is_us_employee', 'is_eu_merchant_data'),
        ('is_merchant_data', 'off_hours'),
        ('distance_km', 'is_merchant_data')
    ])
    
    # Model evaluation
    enable_feature_importance: bool = True
    enable_shap_analysis: bool = False  # Disabled by default for performance
    save_model_artifacts: bool = True

class GeoDataConfig:
    """Optimized geographic data configuration for Amazon offices and data centers"""
    
    @staticmethod
    def get_eu_countries() -> Dict[str, Dict[str, float]]:
        """EU countries with enhanced GDPR strictness scores"""
        return {
            # Tier 1: Strictest enforcement
            'Germany': {'lat': 51.1657, 'lon': 10.4515, 'gdpr_strict': 1.0},
            'France': {'lat': 46.2276, 'lon': 2.2137, 'gdpr_strict': 1.0},
            'Ireland': {'lat': 53.1424, 'lon': -7.6921, 'gdpr_strict': 1.0},
            'Netherlands': {'lat': 52.1326, 'lon': 5.2913, 'gdpr_strict': 1.0},
            'Austria': {'lat': 47.5162, 'lon': 14.5501, 'gdpr_strict': 1.0},
            
            # Tier 2: High enforcement
            'Sweden': {'lat': 60.1282, 'lon': 18.6435, 'gdpr_strict': 0.9},
            'Denmark': {'lat': 56.2639, 'lon': 9.5018, 'gdpr_strict': 0.9},
            'Belgium': {'lat': 50.5039, 'lon': 4.4699, 'gdpr_strict': 0.9},
            'Finland': {'lat': 61.9241, 'lon': 25.7482, 'gdpr_strict': 0.9},
            
            # Tier 3: Moderate enforcement
            'Spain': {'lat': 40.4637, 'lon': -3.7492, 'gdpr_strict': 0.8},
            'Italy': {'lat': 41.8719, 'lon': 12.5674, 'gdpr_strict': 0.8},
            'Poland': {'lat': 51.9194, 'lon': 19.1451, 'gdpr_strict': 0.7},
            'Czech Republic': {'lat': 49.8175, 'lon': 15.4730, 'gdpr_strict': 0.7},
            'Portugal': {'lat': 39.3999, 'lon': -8.2245, 'gdpr_strict': 0.8}
        }
    
    @staticmethod
    def get_data_centers() -> Dict[str, Dict[str, any]]:
        """Enhanced data center locations with capacity and compliance info"""
        return {
            # EU Data Centers
            'eu-west-1': {
                'lat': 53.3498, 'lon': -6.2603, 'country': 'Ireland', 'region': 'EU',
                'capacity': 'high', 'gdpr_compliant': True, 'tier': 1
            },
            'eu-central-1': {
                'lat': 50.1109, 'lon': 8.6821, 'country': 'Germany', 'region': 'EU',
                'capacity': 'high', 'gdpr_compliant': True, 'tier': 1
            },
            'eu-west-2': {
                'lat': 51.5074, 'lon': -0.1278, 'country': 'UK', 'region': 'EU',
                'capacity': 'medium', 'gdpr_compliant': True, 'tier': 2
            },
            'eu-north-1': {
                'lat': 59.3293, 'lon': 18.0686, 'country': 'Sweden', 'region': 'EU',
                'capacity': 'medium', 'gdpr_compliant': True, 'tier': 2
            },
            
            # Non-EU Data Centers
            'us-east-1': {
                'lat': 39.0458, 'lon': -77.5081, 'country': 'USA', 'region': 'US',
                'capacity': 'high', 'gdpr_compliant': False, 'tier': 1
            },
            'us-west-2': {
                'lat': 45.5152, 'lon': -122.6784, 'country': 'USA', 'region': 'US',
                'capacity': 'high', 'gdpr_compliant': False, 'tier': 1
            },
            'ap-southeast-1': {
                'lat': 1.3521, 'lon': 103.8198, 'country': 'Singapore', 'region': 'APAC',
                'capacity': 'medium', 'gdpr_compliant': False, 'tier': 2
            },
            'ap-northeast-1': {
                'lat': 35.6762, 'lon': 139.6503, 'country': 'Japan', 'region': 'APAC',
                'capacity': 'high', 'gdpr_compliant': False, 'tier': 1
            }
        }
    
    @staticmethod
    def get_employee_locations() -> List[Dict[str, any]]:
        """Amazon office locations with realistic employee distribution"""
        return [
            {
                'city': 'Seattle', 'lat': 47.6062, 'lon': -122.3321, 'country': 'USA',
                'weight': 0.30, 'timezone': 'America/Los_Angeles', 'gdpr_jurisdiction': False,
                'office_type': 'headquarters'
            },
            {
                'city': 'Arlington', 'lat': 38.8816, 'lon': -77.0910, 'country': 'USA',
                'weight': 0.25, 'timezone': 'America/New_York', 'gdpr_jurisdiction': False,
                'office_type': 'hq2'
            },
            {
                'city': 'Austin', 'lat': 30.2672, 'lon': -97.7431, 'country': 'USA',
                'weight': 0.15, 'timezone': 'America/Chicago', 'gdpr_jurisdiction': False,
                'office_type': 'tech_hub'
            },
            {
                'city': 'Dublin', 'lat': 53.3498, 'lon': -6.2603, 'country': 'Ireland',
                'weight': 0.12, 'timezone': 'Europe/Dublin', 'gdpr_jurisdiction': True,
                'office_type': 'eu_headquarters'
            },
            {
                'city': 'Berlin', 'lat': 52.5200, 'lon': 13.4050, 'country': 'Germany',
                'weight': 0.08, 'timezone': 'Europe/Berlin', 'gdpr_jurisdiction': True,
                'office_type': 'eu_tech_center'
            },
            {
                'city': 'London', 'lat': 51.5074, 'lon': -0.1278, 'country': 'UK',
                'weight': 0.06, 'timezone': 'Europe/London', 'gdpr_jurisdiction': True,
                'office_type': 'eu_operations'
            },
            {
                'city': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'country': 'India',
                'weight': 0.04, 'timezone': 'Asia/Kolkata', 'gdpr_jurisdiction': False,
                'office_type': 'development_center'
            }
        ]

class ComplianceConfig:
    """Enhanced GDPR compliance rules for Amazon-EU merchant scenario"""
    
    @staticmethod
    def get_country_mapping() -> Dict[str, Dict[str, any]]:
        """Country information for Amazon-EU merchant compliance checks"""
        return {
            # EU Countries (GDPR applies)
            'Ireland': {'is_eu': True, 'strictness': 1.0, 'data_localization': True, 'amazon_office': True},
            'Germany': {'is_eu': True, 'strictness': 1.0, 'data_localization': True, 'amazon_office': True},
            'UK': {'is_eu': True, 'strictness': 0.9, 'data_localization': True, 'amazon_office': True},
            'Netherlands': {'is_eu': True, 'strictness': 1.0, 'data_localization': True, 'amazon_office': False},
            'France': {'is_eu': True, 'strictness': 1.0, 'data_localization': True, 'amazon_office': False},
            'Sweden': {'is_eu': True, 'strictness': 0.9, 'data_localization': True, 'amazon_office': False},
            'Spain': {'is_eu': True, 'strictness': 0.8, 'data_localization': True, 'amazon_office': False},
            'Italy': {'is_eu': True, 'strictness': 0.8, 'data_localization': True, 'amazon_office': False},
            'Poland': {'is_eu': True, 'strictness': 0.7, 'data_localization': True, 'amazon_office': False},
            'Belgium': {'is_eu': True, 'strictness': 0.9, 'data_localization': True, 'amazon_office': False},
            'Austria': {'is_eu': True, 'strictness': 0.9, 'data_localization': True, 'amazon_office': False},
            
            # Non-EU Countries (Amazon offices)
            'USA': {'is_eu': False, 'strictness': 0.3, 'data_localization': False, 'amazon_office': True},
            'India': {'is_eu': False, 'strictness': 0.4, 'data_localization': True, 'amazon_office': True},
            'Singapore': {'is_eu': False, 'strictness': 0.5, 'data_localization': False, 'amazon_office': False},
            'Japan': {'is_eu': False, 'strictness': 0.6, 'data_localization': False, 'amazon_office': False},
            'Canada': {'is_eu': False, 'strictness': 0.4, 'data_localization': False, 'amazon_office': False},
            'Brazil': {'is_eu': False, 'strictness': 0.3, 'data_localization': True, 'amazon_office': False},
        }
    
    @staticmethod
    def get_violation_probabilities() -> Dict[str, float]:
        """Violation probability thresholds for Amazon-EU merchant scenario"""
        return {
            # PRIMARY VIOLATIONS (definite)
            'us_employee_eu_merchant_data': 1.0,  # Definite GDPR violation
            
            # SECONDARY VIOLATIONS (probabilistic)
            'off_hours_merchant_access': 0.4,     # 40% probability
            'suspicious_distance_merchant': 0.3,   # 30% probability
            'weekend_merchant_access': 0.2,        # 20% probability
            'combined_risk_factors': 0.6,          # 60% for multiple risks
            
            # RISK THRESHOLDS
            'high_risk_threshold': 0.75,
            'medium_risk_threshold': 0.5,
            'low_risk_threshold': 0.25,
            
            # BASE PROBABILITIES
            'eu_merchant_data_probability': 0.4,   # 40% of merchant queries are EU
            'merchant_data_probability': 0.35,     # 35% of queries involve merchant data
            'violation_base_rate': 0.12            # 12% baseline violation rate
        }
    
    @staticmethod
    def get_risk_weights() -> Dict[str, float]:
        """Risk factor weights for Amazon-EU merchant scenario"""
        return {
            'cross_border_data_risk': 0.5,     # US employee accessing EU merchant data
            'temporal_risk': 0.2,              # Off-hours/weekend access
            'data_sensitivity_risk': 0.2,      # Merchant data sensitivity
            'geographic_distance_risk': 0.1    # Geographic distance factor
        }
    
    @staticmethod
    def get_business_context():
        """Business context for Amazon-EU merchant GDPR scenario"""
        return {
            'scenario': 'Amazon US employees accessing EU third-party merchant data',
            'violation_type': 'Cross-border data transfer violating EU data residency',
            'regulatory_framework': 'GDPR (General Data Protection Regulation)',
            'key_risk': 'US-based Amazon employees accessing EU merchant customer/transaction data',
            'compliance_requirement': 'EU merchant data must remain within EU jurisdiction',
            'business_impact': 'Potential GDPR fines up to 4% of global annual revenue',
            'mitigation_strategies': [
                'Implement geo-blocking for EU merchant data access from US',
                'Require explicit approval for cross-border merchant data access',
                'Establish EU-only data processing for EU merchants',
                'Monitor and alert on suspicious cross-border data patterns'
            ]
        }