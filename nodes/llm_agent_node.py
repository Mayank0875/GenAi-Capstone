from typing import TYPE_CHECKING
from rag.retriever import retrieve_context
from rag.advisor import generate_analysis

if TYPE_CHECKING:
    from agent_graph import HouseState


def llm_agent_node(state: "HouseState") -> "HouseState":
    """Retrieves EDA context and generates LLM analysis."""
    if state.get("error"):
        return state

    prop  = state["property_input"]
    query = (
        f"price prediction {prop.get('city', '')} "
        f"{prop.get('bedrooms', '')} bedrooms "
        f"{prop.get('sqft_living', '')} sqft "
        f"waterfront={prop.get('waterfront', 0)} "
        f"condition={prop.get('condition', 3)}"
    )

    eda_context = retrieve_context(query)

    try:
        analysis = generate_analysis(
            property_details=prop,
            predicted_price=state["predicted_price"],
            eda_context=eda_context,
        )
        return {**state, "eda_context": eda_context, "analysis": analysis}
    except Exception as e:
        return {**state, "error": f"LLM analysis failed: {str(e)}"}
