# 🏠 House Price Prediction & AI Advisor

An end-to-end system that predicts house prices using ML and provides AI-based reasoning, recommendations, and interactive chat — combining **XGBoost + LangGraph + Groq LLM + RAG**.

---

## 🎯 Objectives

- Predict house prices accurately using XGBoost
- Generate AI explanations and investment recommendations
- Use EDA insights as a RAG knowledge base
- Enable follow-up queries through a chatbot

---

## ⚙️ Workflow

```
User Input → ML Model (XGBoost) → RAG (EDA Insights) → LLM (Groq) → Analysis + Chat
```

---

## 🛠 Tech Stack

| | |
|---|---|
| ML | XGBoost, Scikit-learn |
| Agent | LangGraph, LangChain |
| LLM | Groq API (llama-3.1-8b-instant) |
| UI | Streamlit |
| Data | Pandas, NumPy |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.776 |
| RMSE | ~$102,000 |

---

## 📁 Project Structure

```
├── nodes/               # LangGraph nodes
│   ├── input_node.py
│   ├── prediction_node.py
│   └── llm_agent_node.py
├── rag/                 # RAG module
│   ├── eda_context.py
│   ├── retriever.py
│   └── advisor.py
├── components/          # ML pipeline
├── agent_graph.py       # Pipeline assembly
├── app.py               # Streamlit app
├── model.py             # Model training
└── house_price_model.pkl
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Train model (if needed)
python model.py

# Run app
streamlit run app.py
```

---

## ✨ Features

- ML-based price prediction
- AI-powered analysis and investment advice
- RAG using EDA insights
- LangGraph 3-node agent pipeline
- Interactive chatbot with memory

---

## 👤 Author

**Mayank Gupta** — Enrollment No: 2401010267
