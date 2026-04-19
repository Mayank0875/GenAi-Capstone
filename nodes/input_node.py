"""
nodes/input_node.py
-------------------
Node 1 — validates and normalises user property input.
Adds derived features (total_sqft, bath_per_bed) expected by the ML model.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_graph import HouseState

REQUIRED_FIELDS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement", "house_age",
    "waterfront", "view", "condition", "city", "statezip",
    "has_been_renovated",
]


def input_node(state: "HouseState") -> "HouseState":
    """Validates required fields and derives computed features."""
    data    = state["property_input"]
    missing = [f for f in REQUIRED_FIELDS if f not in data]

    if missing:
        return {**state, "error": f"Missing required fields: {missing}"}

    data.setdefault("total_sqft",   int(data["sqft_living"]) + int(data["sqft_basement"]))
    data.setdefault("bath_per_bed", float(data["bathrooms"]) / (float(data["bedrooms"]) + 1))

    return {**state, "property_input": data, "error": ""}
