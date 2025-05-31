"""
High-Performance Optimization Functions using Numba JIT Compilation
Enhanced with CORRECTED Amazon-EU merchant GDPR compliance logic
"""

import numpy as np
import pandas as pd
from numba import njit, jit, prange, types
from numba.typed import Dict
from typing import Tuple, Optional, Union
import warnings

# Suppress numba warnings for cleaner output
try:
    from numba.core.errors import NumbaWarning
    warnings.filterwarnings('ignore', category=NumbaWarning)
except ImportError:
    warnings.filterwarnings('ignore')

# =============================================================================
# GEOGRAPHIC CALCULATIONS - Ultra High Performance
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def haversine_distance_vectorized(lat1: np.ndarray, lon1: np.ndarray, 
                                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Ultra-fast vectorized haversine distance calculation using Numba JIT
    
    Parameters:
    -----------
    lat1, lon1 : ndarray
        Latitude and longitude of first points in degrees
    lat2, lon2 : ndarray  
        Latitude and longitude of second points in degrees
        
    Returns:
    --------
    distances : ndarray
        Distances in kilometers
        
    Performance: ~100x faster than pure Python, ~10x faster than pandas
    """
    R = 6371.0  # Earth's radius in km (using float for better performance)
    n = lat1.shape[0]
    distances = np.empty(n, dtype=np.float32)
    
    for i in prange(n):  # Parallel execution
        # Convert to radians (optimized)
        lat1_rad = np.radians(lat1[i])
        lon1_rad = np.radians(lon1[i])
        lat2_rad = np.radians(lat2[i])
        lon2_rad = np.radians(lon2[i])
        
        # Haversine formula (optimized)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Use fast math operations
        sin_dlat_2 = np.sin(dlat * 0.5)
        sin_dlon_2 = np.sin(dlon * 0.5)
        
        a = (sin_dlat_2 * sin_dlat_2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * sin_dlon_2 * sin_dlon_2)
        
        # Optimized arc calculation
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        distances[i] = R * c
    
    return distances

@njit(cache=True, fastmath=True)
def haversine_distance_single(lat1: float, lon1: float, 
                             lat2: float, lon2: float) -> float:
    """
    Optimized single-point haversine distance calculation
    
    Returns:
    --------
    distance : float
        Distance in kilometers
    """
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    sin_dlat_2 = np.sin(dlat * 0.5)
    sin_dlon_2 = np.sin(dlon * 0.5)
    
    a = (sin_dlat_2 * sin_dlat_2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * sin_dlon_2 * sin_dlon_2)
    
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    return R * c

# =============================================================================
# CORRECTED GDPR COMPLIANCE LOGIC - Amazon-EU Merchant Scenario
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def violation_logic_vectorized(is_us_employee: np.ndarray, is_eu_merchant_data: np.ndarray, 
                              is_merchant_data: np.ndarray, off_hours: np.ndarray,
                              suspicious_distance: np.ndarray, random_vals: np.ndarray,
                              strictness_scores: Optional[np.ndarray] = None) -> np.ndarray:
    """
    CORRECTED: Amazon-EU merchant GDPR violation detection
    
    REAL SCENARIO: US Amazon employees accessing EU merchant data violates GDPR
    
    Parameters:
    -----------
    is_us_employee : ndarray (int8)
        Boolean array indicating US-based Amazon employees
    is_eu_merchant_data : ndarray (int8)
        Boolean array indicating EU merchant data (subject to GDPR)
    is_merchant_data : ndarray (int8)
        Boolean array indicating any merchant data access
    off_hours : ndarray (int8)
        Boolean array indicating off-hours access
    suspicious_distance : ndarray (int8)
        Boolean array indicating suspicious geographic distance
    random_vals : ndarray (float32)
        Random values for probabilistic violation logic
    strictness_scores : ndarray (float32), optional
        GDPR strictness scores by merchant's country
        
    Returns:
    --------
    violations : ndarray (int8)
        Binary array indicating violations (1 = violation, 0 = no violation)
    """
    n = is_us_employee.shape[0]
    violations = np.zeros(n, dtype=np.int8)
    
    # Violation thresholds for secondary risks
    off_hours_threshold = 0.4
    distance_threshold = 0.3
    combined_risk_threshold = 0.6
    
    for i in prange(n):
        violation_score = 0.0
        
        # PRIMARY GDPR VIOLATION: US Amazon employee accessing EU merchant data
        # This violates EU data residency and GDPR requirements
        if is_us_employee[i] and is_eu_merchant_data[i]:
            # Apply merchant's country GDPR strictness if available
            if strictness_scores is not None:
                violation_score = strictness_scores[i]
            else:
                violation_score = 1.0  # Definite violation
        
        # SECONDARY VIOLATIONS: Additional risk factors for any merchant data
        elif is_merchant_data[i]:
            # Off-hours access to merchant data (suspicious pattern)
            if off_hours[i] and random_vals[i] < off_hours_threshold:
                violation_score = 0.7
            
            # Suspicious geographic distance for merchant data access
            elif suspicious_distance[i] and random_vals[i] < distance_threshold:
                violation_score = 0.6
            
            # Combined risk factors (multiplicative effect)
            elif off_hours[i] and suspicious_distance[i]:
                if random_vals[i] < (off_hours_threshold * distance_threshold):
                    violation_score = 0.8
        
        # Apply final violation decision
        if violation_score >= 0.5:
            violations[i] = 1
    
    return violations

@njit(cache=True, fastmath=True, parallel=True)
def enhanced_violation_scoring(is_us_employee: np.ndarray, is_eu_merchant_data: np.ndarray,
                              is_merchant_data: np.ndarray, off_hours: np.ndarray,
                              suspicious_distance: np.ndarray, weekend: np.ndarray,
                              distance_km: np.ndarray, hour: np.ndarray,
                              merchant_country_strictness: np.ndarray) -> np.ndarray:
    """
    CORRECTED: Advanced violation scoring for Amazon-EU merchant scenario
    
    Returns:
    --------
    risk_scores : ndarray (float32)
        Continuous risk scores (0.0 to 1.0)
    """
    n = is_us_employee.shape[0]
    risk_scores = np.zeros(n, dtype=np.float32)
    
    for i in prange(n):
        score = 0.0
        
        # PRIMARY RISK: US employee accessing EU merchant data (50% weight)
        if is_us_employee[i] and is_eu_merchant_data[i]:
            # Scale by merchant's country GDPR strictness
            gdpr_factor = merchant_country_strictness[i] if merchant_country_strictness[i] > 0 else 1.0
            score += 0.5 * gdpr_factor
        
        # DATA SENSITIVITY RISK: Any merchant data access (25% weight)
        if is_merchant_data[i]:
            score += 0.25
            
            # Additional risk if it's EU merchant data
            if is_eu_merchant_data[i]:
                score += 0.1  # Extra 10% for EU merchant sensitivity
        
        # TEMPORAL RISK: Time-based suspicious patterns (15% weight)
        if off_hours[i]:
            score += 0.1
        if weekend[i]:
            score += 0.05
        
        # GEOGRAPHIC RISK: Cross-border distance factors (10% weight)
        if suspicious_distance[i]:
            # Scale by actual distance (normalized to max 20,000km)
            distance_factor = min(distance_km[i] / 20000.0, 1.0)
            score += 0.1 * distance_factor
        
        # HOUR-SPECIFIC RISK: Very late/early hours indicate potential unauthorized access
        if 2 <= hour[i] <= 5:  # 2AM-5AM most suspicious
            score += 0.05
        elif hour[i] <= 1 or hour[i] >= 23:  # Late night/early morning
            score += 0.03
        
        risk_scores[i] = min(score, 1.0)  # Cap at 1.0
    
    return risk_scores

# =============================================================================
# TIME-BASED FEATURE CALCULATIONS
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def calculate_time_features(hours: np.ndarray, days_of_week: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced time-based feature calculation with business logic
    
    Parameters:
    -----------
    hours : ndarray (int8)
        Hour of day (0-23)
    days_of_week : ndarray (int8)
        Day of week (0-6, where 0=Monday)
        
    Returns:
    --------
    off_hours : ndarray (int8)
        Enhanced off-hours detection (outside 6 AM - 10 PM)
    weekend : ndarray (int8)
        Weekend detection (Saturday=5, Sunday=6)
    """
    n = hours.shape[0]
    off_hours = np.zeros(n, dtype=np.int8)
    weekend = np.zeros(n, dtype=np.int8)
    
    for i in prange(n):
        # Business hours: 6 AM - 10 PM (Amazon operates globally)
        if hours[i] < 6 or hours[i] > 22:
            off_hours[i] = 1
        
        # Weekend detection (Saturday=5, Sunday=6)
        if days_of_week[i] >= 5:
            weekend[i] = 1
    
    return off_hours, weekend

# =============================================================================
# RISK ASSESSMENT AND GEOGRAPHIC ANALYSIS
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def calculate_risk_factors(distances: np.ndarray, distance_threshold: float = 5000.0,
                          extreme_distance_threshold: float = 10000.0) -> np.ndarray:
    """
    Enhanced geographic risk factor calculation with multiple thresholds
    
    For Amazon-EU scenario:
    - Normal: <5000km (e.g., US East Coast to US West Coast)
    - Suspicious: 5000-10000km (e.g., US to Europe distances)
    - Extreme: >10000km (e.g., US to Asia)
    
    Parameters:
    -----------
    distances : ndarray (float32)
        Distances in kilometers
    distance_threshold : float
        Threshold for suspicious distance (default 5000 km)
    extreme_distance_threshold : float
        Threshold for extreme distance (default 10000 km)
        
    Returns:
    --------
    risk_levels : ndarray (int8)
        Risk levels: 0=normal, 1=suspicious, 2=extreme
    """
    n = distances.shape[0]
    risk_levels = np.zeros(n, dtype=np.int8)
    
    for i in prange(n):
        if distances[i] > extreme_distance_threshold:
            risk_levels[i] = 2  # Extreme risk (>10,000km)
        elif distances[i] > distance_threshold:
            risk_levels[i] = 1  # Suspicious risk (5,000-10,000km)
        else:
            risk_levels[i] = 0  # Normal risk (<5,000km)
    
    return risk_levels

# =============================================================================
# BATCH PROCESSING AND OPTIMIZATION
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def batch_process_violations(employee_countries: np.ndarray, merchant_countries: np.ndarray,
                           merchant_data: np.ndarray, off_hours: np.ndarray,
                           suspicious_distances: np.ndarray, 
                           random_vals: np.ndarray,
                           batch_size: int = 10000) -> np.ndarray:
    """
    CORRECTED: High-performance batch processing for Amazon-EU merchant violations
    
    Parameters:
    -----------
    employee_countries : ndarray (int8)
        Employee country codes (0=EU, 1=US, 2=Other)
    merchant_countries : ndarray (int8)
        Merchant country codes (0=EU, 1=US, 2=Other)
    merchant_data : ndarray (int8)
        Merchant data access flags
    off_hours : ndarray (int8)
        Off-hours access flags
    suspicious_distances : ndarray (int8)
        Suspicious distance flags
    random_vals : ndarray (float32)
        Random values for probabilistic logic
    batch_size : int
        Processing batch size for memory efficiency
        
    Returns:
    --------
    violations : ndarray (int8)
        Violation detection results
    """
    n_records = employee_countries.shape[0]
    violations = np.zeros(n_records, dtype=np.int8)
    
    # Process in batches for memory efficiency
    for batch_start in range(0, n_records, batch_size):
        batch_end = min(batch_start + batch_size, n_records)
        
        # Process batch in parallel
        for i in prange(batch_start, batch_end):
            is_us_employee = employee_countries[i] == 1  # US Amazon employee
            is_eu_merchant = merchant_countries[i] == 0  # EU merchant data
            
            # PRIMARY VIOLATION: US employee accessing EU merchant data
            if is_us_employee and is_eu_merchant and merchant_data[i]:
                violations[i] = 1
            # SECONDARY VIOLATIONS: Suspicious patterns with merchant data
            elif merchant_data[i]:
                if off_hours[i] and random_vals[i] < 0.4:
                    violations[i] = 1
                elif suspicious_distances[i] and random_vals[i] < 0.3:
                    violations[i] = 1
    
    return violations

# =============================================================================
# UTILITY AND HELPER FUNCTIONS
# =============================================================================

def optimize_arrays_for_numba(df: pd.DataFrame) -> dict:
    """
    Convert DataFrame columns to optimized NumPy arrays for Numba processing
    
    Parameters:
    -----------
    df : DataFrame
        Input pandas DataFrame
        
    Returns:
    --------
    arrays : dict
        Dictionary of optimized NumPy arrays
    """
    arrays = {}
    
    # Convert columns to appropriate dtypes
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert to categorical codes
            arrays[col] = df[col].astype('category').cat.codes.values.astype(np.int32)
        elif df[col].dtype in ['int64', 'int32']:
            arrays[col] = df[col].values.astype(np.int32)
        elif df[col].dtype in ['float64', 'float32']:
            arrays[col] = df[col].values.astype(np.float32)
        elif df[col].dtype == 'bool':
            arrays[col] = df[col].values.astype(np.int8)
        else:
            arrays[col] = df[col].values
    
    return arrays

# Performance benchmarking decorator
def benchmark_numba_function(func):
    """
    Decorator to benchmark Numba function performance
    """
    def wrapper(*args, **kwargs):
        import time
        
        # Warm up JIT compilation
        if hasattr(func, 'py_func'):
            func(*args, **kwargs)
        
        # Benchmark
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        print(f"⚡ {func.__name__} executed in {(end_time - start_time)*1000:.2f}ms")
        return result
    
    return wrapper