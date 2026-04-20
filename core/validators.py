from typing import Dict, Any, List

def validate_input(data: Dict[str, Any]) -> List[str]:
    errors = []
    if data["carpet_area"] <= 0:                    errors.append("Carpet area must be > 0.")
    if data["carpet_area"] > 50_000:                errors.append("Carpet area too large (> 50,000 sq ft).")
    if not (1 <= data["num_rooms"] <= 20):           errors.append("Rooms must be 1–20.")
    if not (1 <= data["num_bathrooms"] <= 15):       errors.append("Bathrooms must be 1–15.")
    if data["num_bathrooms"] > data["num_rooms"]:    errors.append("Bathrooms cannot exceed rooms.")
    if not (0.0 <= data["property_tax_rate"] <= 10): errors.append("Tax rate must be 0–10%.")
    if data["Estimated Value"] <= 0:                 errors.append("Estimated value must be positive.")
    if not (1900 <= data["Year"] <= 2030):           errors.append("Year must be 1900–2030.")
    return errors
