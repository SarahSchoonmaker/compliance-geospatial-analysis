"""
GDPR Compliance Analysis Package
High-performance modular architecture
"""

__version__ = "2.0.0"
__author__ = "GDPR Compliance Team"

# Import all key components for easy access
from .config import PerformanceConfig, DataConfig, ModelConfig, GeoDataConfig, ComplianceConfig
from .performance import PerformanceMonitor, SystemInfo, performance_warning
from .optimization import (
    haversine_distance_vectorized, 
    violation_logic_vectorized, 
    calculate_time_features, 
    calculate_risk_factors,
    batch_process_violations
)
from .geographic_setup import GeographicDataSetup
from .data_generator import SyntheticDataGenerator
from .model_trainer import ComplianceModelTrainer
from .dashboard_creator import DashboardCreator
from .report_generator import ReportGenerator

__all__ = [
    # Configuration classes
    'PerformanceConfig', 'DataConfig', 'ModelConfig', 'GeoDataConfig', 'ComplianceConfig',
    
    # Performance monitoring
    'PerformanceMonitor', 'SystemInfo', 'performance_warning',
    
    # Optimization functions
    'haversine_distance_vectorized', 'violation_logic_vectorized', 
    'calculate_time_features', 'calculate_risk_factors', 'batch_process_violations',
    
    # Main components
    'GeographicDataSetup', 'SyntheticDataGenerator', 'ComplianceModelTrainer',
    'DashboardCreator', 'ReportGenerator'
]