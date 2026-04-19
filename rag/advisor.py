"""
rag/advisor.py
--------------
LangChain + Groq LLM chains:
  - generate_analysis()  → full property analysis + investment recommendation
  - answer_followup()    → conversational follow-up with chat history
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.4, api_key=api_key)


_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert real estate analyst. Use the provided EDA market context to ground "
     "your analysis in real data. Be concise, specific, and actionable."),
    ("human",
     """PROPERTY DETAILS:
{property_details}

PREDICTED PRICE: ${predicted_price:,.0f}

MARKET CONTEXT (from EDA):
{eda_context}

Please provide:
1. **Price Analysis** – Is this price fair, above, or below market? Why?
2. **Property Strengths** – What features add the most value?
3. **Investment Recommendation** – Buy / Sell / Hold / Invest? Justify briefly.
4. **Key Risks** – Any concerns about this property?
"""),
])

_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful real estate assistant with access to a predicted house price, "
     "EDA market insights, and the conversation history. Be concise and helpful."),
    ("human",
     """PROPERTY DETAILS: {property_details}
PREDICTED PRICE: ${predicted_price:,.0f}
MARKET CONTEXT: {eda_context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}
"""),
])


def generate_analysis(property_details: dict, predicted_price: float, eda_context: str) -> str:
    """Generates a full property analysis with investment recommendation."""
    chain = _ANALYSIS_PROMPT | _get_llm() | StrOutputParser()
    return chain.invoke({
        "property_details": "\n".join(f"  {k}: {v}" for k, v in property_details.items()),
        "predicted_price":  predicted_price,
        "eda_context":      eda_context,
    })


def answer_followup(
    question: str,
    property_details: dict,
    predicted_price: float,
    eda_context: str,
    chat_history: list[dict],
) -> str:
    """Answers a follow-up question using property context and chat history."""
    chain = _CHAT_PROMPT | _get_llm() | StrOutputParser()
    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history[-6:]
    )
    return chain.invoke({
        "property_details": "\n".join(f"  {k}: {v}" for k, v in property_details.items()),
        "predicted_price":  predicted_price,
        "eda_context":      eda_context,
        "chat_history":     history_str or "No prior conversation.",
        "question":         question,
    })
