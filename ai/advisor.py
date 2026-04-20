import streamlit as st
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

@st.cache_resource
def get_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        logger.error(f"LLM init failed: {e}")
        return None

def generate_advice(input_data, price, market, llm) -> str:
    if llm is None: return "AI advisor unavailable."
    prop = (
        f"Carpet Area: {input_data.get('carpet_area','N/A')} sq ft | "
        f"Rooms: {input_data.get('num_rooms','N/A')} | "
        f"Bathrooms: {input_data.get('num_bathrooms','N/A')} | "
        f"Tax Rate: {input_data.get('property_tax_rate','N/A')}% | "
        f"Estimated Value: Rs {input_data.get('Estimated Value',0):,.0f} | "
        f"Year: {input_data.get('Year','N/A')}"
    )
    mkt = " | ".join(market[:3])
    prompt = (
        "You are a certified real estate investment advisor in India. "
        "Only answer about real estate. Do not invent price figures. "
        f"Property: {prop}\nPredicted Price: Rs {price:,.0f}\nMarket: {mkt}\n\n"
        "Provide:\n1. Valuation Summary\n2. Recommendation (Buy/Hold/Avoid)\n"
        "3. Key Risk Factors\n4. Final Advice\n5. Disclaimer"
    )
    try:
        tokenizer = llm["tokenizer"]
        model = llm["model"]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=300)
        advice = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        if not advice or len(advice) < 20:
            return "AI advisor could not generate a response for these inputs."
        if any(p in advice.lower() for p in ["ignore previous","disregard","jailbreak"]):
            return "Suspicious output detected. Please try again."
        return advice
    except Exception as e:
        return f"Advice generation failed: {e}"
