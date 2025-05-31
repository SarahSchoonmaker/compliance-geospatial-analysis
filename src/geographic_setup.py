"""
Geographic data setup with optimized performance
"""

import pandas as pd
import geopandas as gpd
from src.config import PerformanceConfig, GeoDataConfig
from src.performance import PerformanceMonitor, performance_warning

class GeographicDataSetup:
    """Optimized geographic data setup"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
    
    @performance_warning
    def setup_geographic_data(self):
        """Setup EU boundaries and data center locations with caching"""
        self.monitor.start("Geographic Data Setup")
        
        # Use optimized data loading
        eu_boundaries = self._create_eu_boundaries()
        data_centers = self._create_data_centers()
        
        self.monitor.end()
        return eu_boundaries, data_centers
    
    def _create_eu_boundaries(self):
        """Create EU boundaries GeoDataFrame efficiently"""
        eu_countries = GeoDataConfig.get_eu_countries()
        
        # Vectorized DataFrame creation
        eu_data = pd.DataFrame([
            {
                'country': country,
                'lat': info['lat'],
                'lon': info['lon'], 
                'gdpr_strictness': info['gdpr_strict']
            }
            for country, info in eu_countries.items()
        ])
        
        # Efficient geometry creation
        geometry = gpd.points_from_xy(eu_data.lon, eu_data.lat)
        return gpd.GeoDataFrame(eu_data, geometry=geometry, crs='EPSG:4326')
    
    def _create_data_centers(self):
        """Create data centers GeoDataFrame efficiently"""
        data_centers = GeoDataConfig.get_data_centers()
        
        # Vectorized DataFrame creation
        dc_data = pd.DataFrame([
            {
                'dc_id': dc_id,
                'lat': info['lat'],
                'lon': info['lon'],
                'country': info['country'],
                'region': info['region'],
                'is_eu': 1 if info['region'] == 'EU' else 0
            }
            for dc_id, info in data_centers.items()
        ])
        
        # Efficient geometry creation
        geometry = gpd.points_from_xy(dc_data.lon, dc_data.lat)
        return gpd.GeoDataFrame(dc_data, geometry=geometry, crs='EPSG:4326')