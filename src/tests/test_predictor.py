print("Testing environment setup...")
try:
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import folium
    import sklearn
    print("✅ All packages installed successfully!")
    print(f"Python location: {pd.__file__}")
except ImportError as e:
    print(f"❌ Missing package: {e}")