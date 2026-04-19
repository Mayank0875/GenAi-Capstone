from rag.eda_context import EDA_CONTEXT


def _parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key   = "general"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("---") and line.endswith("---"):
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key   = line.strip("- ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


_SECTIONS = _parse_sections(EDA_CONTEXT)

_KEYWORD_MAP: dict[str, list[str]] = {
    "price":     ["Dataset Overview", "Pricing Tiers", "Key Feature Statistics"],
    "bedroom":   ["Key Feature Statistics", "Feature Correlations with Price (Pearson)"],
    "bathroom":  ["Key Feature Statistics", "Feature Correlations with Price (Pearson)"],
    "sqft":      ["Key Feature Statistics", "Feature Correlations with Price (Pearson)"],
    "size":      ["Key Feature Statistics", "Feature Correlations with Price (Pearson)"],
    "waterfront":["Market Insights"],
    "renovate":  ["Market Insights"],
    "view":      ["View Rating Impact on Price"],
    "condition": ["Condition Rating Impact on Price"],
    "city":      ["Top 5 Cities by Median Price"],
    "location":  ["Top 5 Cities by Median Price"],
    "invest":    ["Pricing Tiers", "Market Insights"],
    "buy":       ["Pricing Tiers", "Market Insights"],
    "sell":      ["Pricing Tiers", "Market Insights"],
    "age":       ["Key Feature Statistics", "Feature Correlations with Price (Pearson)"],
    "correlat":  ["Feature Correlations with Price (Pearson)"],
}


def retrieve_context(query: str, max_sections: int = 4) -> str:
    """Returns relevant EDA sections for the query; falls back to full context."""
    query_lower     = query.lower()
    matched: set[str] = set()

    for keyword, section_names in _KEYWORD_MAP.items():
        if keyword in query_lower:
            matched.update(section_names)

    if not matched:
        return EDA_CONTEXT

    retrieved = []
    for name in list(matched)[:max_sections]:
        content = _SECTIONS.get(name, "")
        if content:
            retrieved.append(f"--- {name} ---\n{content}")

    return "\n\n".join(retrieved) if retrieved else EDA_CONTEXT


def get_full_context() -> str:
    """Returns the complete EDA knowledge base."""
    return EDA_CONTEXT
