from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from nodes.input_node      import input_node
from nodes.prediction_node import prediction_node
from nodes.llm_agent_node  import llm_agent_node


class HouseState(TypedDict):
    property_input:  dict[str, Any]
    predicted_price: float
    eda_context:     str
    analysis:        str
    error:           str


def _build_graph():
    graph = StateGraph(HouseState)

    graph.add_node("input_node",      input_node)
    graph.add_node("prediction_node", prediction_node)
    graph.add_node("llm_agent_node",  llm_agent_node)

    graph.set_entry_point("input_node")
    graph.add_edge("input_node",      "prediction_node")
    graph.add_edge("prediction_node", "llm_agent_node")
    graph.add_edge("llm_agent_node",  END)

    return graph.compile()


_pipeline = _build_graph()


def run_agent(property_input: dict) -> dict:
    """Runs the full pipeline. Returns predicted_price, analysis, and error."""
    result = _pipeline.invoke({
        "property_input":  property_input,
        "predicted_price": 0.0,
        "eda_context":     "",
        "analysis":        "",
        "error":           "",
    })
    return {
        "predicted_price": result["predicted_price"],
        "analysis":        result["analysis"],
        "error":           result["error"],
    }
