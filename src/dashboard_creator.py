"""
Optimized dashboard creation with memory-efficient visualization
"""

import pandas as pd
import folium
from folium.plugins import HeatMap, FastMarkerCluster

from src.config import PerformanceConfig, DataConfig
from src.performance import PerformanceMonitor, performance_warning

class DashboardCreator:
    """Memory-efficient dashboard creation"""
    
    def __init__(self, config: PerformanceConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self.monitor = PerformanceMonitor()
    
    @performance_warning
    def create_dashboard(self, query_data: pd.DataFrame, data_centers: pd.DataFrame, 
                        eu_boundaries: pd.DataFrame = None):
        """Create optimized interactive dashboard"""
        self.monitor.start("Dashboard Creation")
        
        # Memory-efficient sampling
        dashboard_data = self._sample_data_efficiently(query_data)
        
        # Create base map
        m = folium.Map(location=[50.0, 10.0], zoom_start=4)
        
        # Add layers efficiently
        self._add_eu_boundaries(m, eu_boundaries)
        self._add_data_centers(m, data_centers)
        self._add_violation_heatmap(m, dashboard_data)
        self._add_violation_markers(m, dashboard_data)
        
        # Save with compression
        dashboard_path = 'compliance-dashboard.html'
        m.save(dashboard_path)
        
        violation_count = len(dashboard_data[dashboard_data['violation'] == 1])
        print(f"📊 Dashboard saved: {dashboard_path}")
        print(f"🔍 Violations displayed: {violation_count:,} / {len(dashboard_data):,}")
        
        self.monitor.end()
        return dashboard_path
    
    def _sample_data_efficiently(self, query_data: pd.DataFrame):
        """Efficiently sample data for dashboard performance"""
        max_points = self.data_config.max_dashboard_points
        
        if len(query_data) <= max_points:
            return query_data
        
        # Stratified sampling to ensure violations are represented
        violations = query_data[query_data['violation'] == 1]
        non_violations = query_data[query_data['violation'] == 0]
        
        # Sample proportionally but ensure minimum violation representation
        violation_sample_size = min(len(violations), max_points // 3)
        non_violation_sample_size = max_points - violation_sample_size
        
        violation_sample = violations.sample(
            n=violation_sample_size, 
            random_state=42
        ) if len(violations) > 0 else pd.DataFrame()
        
        non_violation_sample = non_violations.sample(
            n=min(non_violation_sample_size, len(non_violations)), 
            random_state=42
        ) if len(non_violations) > 0 else pd.DataFrame()
        
        return pd.concat([violation_sample, non_violation_sample], ignore_index=True)
    
    def _add_eu_boundaries(self, m: folium.Map, eu_boundaries: pd.DataFrame):
        """Add EU boundaries with efficient rendering"""
        if eu_boundaries is None:
            return
        
        for _, country in eu_boundaries.iterrows():
            color = 'darkgreen' if country['gdpr_strictness'] >= 1.0 else 'orange'
            folium.CircleMarker(
                [country['lat'], country['lon']],
                radius=6,
                popup=f"{country['country']} (Strictness: {country['gdpr_strictness']})",
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
    
    def _add_data_centers(self, m: folium.Map, data_centers: pd.DataFrame):
        """Add data center markers efficiently"""
        for _, dc in data_centers.iterrows():
            color = 'blue' if dc['region'] == 'EU' else 'red'
            folium.Marker(
                [dc['lat'], dc['lon']],
                popup=f"DC: {dc['dc_id']}<br>Country: {dc['country']}<br>Region: {dc['region']}",
                icon=folium.Icon(color=color, icon='server', prefix='fa'),
                tooltip=f"Data Center: {dc['dc_id']}"
            ).add_to(m)
    
    def _add_violation_heatmap(self, m: folium.Map, dashboard_data: pd.DataFrame):
        """Add efficient violation heatmap"""
        violation_data = dashboard_data[dashboard_data['violation'] == 1]
        
        if len(violation_data) > 0:
            heat_data = violation_data[['employee_lat', 'employee_lon']].values.tolist()
            
            HeatMap(
                heat_data,
                radius=15,
                blur=10,
                max_zoom=1,
                gradient={
                    0.4: 'blue', 
                    0.6: 'cyan', 
                    0.7: 'lime', 
                    0.8: 'yellow', 
                    1.0: 'red'
                }
            ).add_to(m)
    
    def _add_violation_markers(self, m: folium.Map, dashboard_data: pd.DataFrame):
        """Add violation markers with clustering for performance"""
        violation_data = dashboard_data[dashboard_data['violation'] == 1]
        
        if len(violation_data) == 0:
            return
        
        # Use clustering for performance with many markers
        if len(violation_data) > 100:
            marker_cluster = FastMarkerCluster(violation_data[['employee_lat', 'employee_lon']].values.tolist())
            marker_cluster.add_to(m)
        else:
            # Add individual markers for smaller datasets
            for _, row in violation_data.iterrows():
                folium.CircleMarker(
                    [row['employee_lat'], row['employee_lon']],
                    radius=4,
                    popup=self._create_violation_popup(row),
                    color='red',
                    fillColor='red',
                    fillOpacity=0.8,
                    weight=1
                ).add_to(m)
    
    def _create_violation_popup(self, row):
        """Create efficient popup text"""
        return f"""
        <b>Violation: {row['query_id']}</b><br>
        Employee: {row['employee_city']}, {row['employee_country']}<br>
        Data Center: {row['data_center']} ({row['dc_country']})<br>
        Distance: {row['distance_km']:.0f}km<br>
        Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}
        """