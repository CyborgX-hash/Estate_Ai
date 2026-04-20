from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    input:           Dict[str, Any]
    predicted_price: float
    market_data:     List[str]
    comps:           List[Dict]
    final_advice:    str
    model_metrics:   Dict[str, float]
    error:           str
