import logging
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_comparable_properties(input_data, df_raw, n=3) -> List[Dict]:
    try:
        required = {"carpet_area","num_rooms","Sale Price"}
        if not required.issubset(df_raw.columns): return []
        df_c  = df_raw.copy()
        area  = input_data.get("carpet_area", 1000)
        rooms = input_data.get("num_rooms", 3)
        mask  = (df_c["carpet_area"] >= area * 0.7) & (df_c["carpet_area"] <= area * 1.3)
        filt  = df_c[mask & (df_c["num_rooms"] == rooms)]
        if len(filt) < n: filt = df_c[mask]
        if filt.empty: return []
        sample = filt.sample(min(n, len(filt)), random_state=42)
        return [{
            "Carpet Area (sq ft)": f"{row.get('carpet_area','N/A'):,.0f}",
            "Rooms":               int(row.get("num_rooms", 0)),
            "Bathrooms":           int(row.get("num_bathrooms", 0)),
            "Sale Price":          f"₹{row.get('Sale Price', 0):,.0f}",
            "Est. Value":          f"₹{row.get('Estimated Value', 0):,.0f}",
        } for _, row in sample.iterrows()]
    except Exception as e:
        logger.error(f"Comps error: {e}"); return []
