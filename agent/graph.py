import streamlit as st
from langgraph.graph import StateGraph, END
from core.state import AgentState
from models.predictor import predict_price
from ai.rag import retrieve_market
from models.comparables import get_comparable_properties
from ai.advisor import generate_advice

def predict_node(state: AgentState) -> AgentState:
    try:
        state["predicted_price"] = predict_price(
            state["input"], st.session_state["model"], st.session_state["feature_columns"])
    except Exception as e:
        state["error"] = f"Prediction: {e}"; state["predicted_price"] = 0.0
    return state

def rag_node(state: AgentState) -> AgentState:
    try:
        q = f"property investment {state['input'].get('carpet_area','')} sqft {state['input'].get('num_rooms','')} rooms India"
        state["market_data"] = retrieve_market(q, st.session_state["vectorstore"])
    except Exception as e:
        state["error"] = f"RAG: {e}"; state["market_data"] = []
    return state

def comps_node(state: AgentState) -> AgentState:
    try:
        state["comps"] = get_comparable_properties(state["input"], st.session_state["df_raw"])
    except Exception as e:
        state["error"] = f"Comps: {e}"; state["comps"] = []
    return state

def advisor_node(state: AgentState) -> AgentState:
    try:
        state["final_advice"] = generate_advice(
            state["input"], state["predicted_price"],
            state["market_data"], st.session_state["llm"])
    except Exception as e:
        state["error"] = f"Advisor: {e}"; state["final_advice"] = "Advisory unavailable."
    return state

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("predict", predict_node)
    g.add_node("rag",     rag_node)
    g.add_node("comps",   comps_node)
    g.add_node("advisor", advisor_node)
    g.set_entry_point("predict")
    g.add_edge("predict","rag")
    g.add_edge("rag","comps")
    g.add_edge("comps","advisor")
    g.add_edge("advisor", END)
    return g.compile()
