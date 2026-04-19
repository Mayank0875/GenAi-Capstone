from typing import TYPE_CHECKING
from predict import predict_house_price

if TYPE_CHECKING:
    from agent_graph import HouseState


def prediction_node(state: "HouseState") -> "HouseState":
    """Calls the ML model and writes predicted_price into state."""
    if state.get("error"):
        return state

    try:
        price = predict_house_price(state["property_input"])
        return {**state, "predicted_price": price}
    except Exception as e:
        return {**state, "error": f"Prediction failed: {str(e)}"}
