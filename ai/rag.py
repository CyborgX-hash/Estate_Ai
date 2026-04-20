import streamlit as st
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

MARKET_CORPUS = [
    "Gurgaon property prices are rising 8% annually driven by IT sector growth.",
    "Gurgaon Golf Course Road has premium residential properties above Rs 10,000 per sq ft.",
    "Gurgaon Dwarka Expressway offers affordable housing with excellent metro connectivity.",
    "Cyber City Gurgaon drives significant commercial real estate demand.",
    "Gurgaon residential rental yields average 3-4% and commercial 6-8% annually.",
    "Noida offers high rental yield of 4-5% for 2BHK apartments.",
    "Noida Sector 150 has emerged as a premium residential destination with sports infrastructure.",
    "Noida Extension Greater Noida West offers affordable housing below Rs 5,000 per sq ft.",
    "Noida metro connectivity boosts adjacent property values by 15-20%.",
    "Noida Authority plots appreciated 40% in three years due to infrastructure expansion.",
    "Delhi real estate is stable but expensive with limited new housing supply.",
    "South Delhi premium micro-markets like Hauz Khas see prices above Rs 25,000 per sq ft.",
    "Delhi Dwarka is a popular middle-income housing destination near the international airport.",
    "Rohini Delhi offers moderate property prices with strong social infrastructure.",
    "East Delhi markets like Laxmi Nagar offer affordable options below Rs 8,000 per sq ft.",
    "Sonipat Haryana property market is growing due to proximity to Delhi and KMP Expressway.",
    "IMT Manesar and Kundli Industrial clusters are boosting Sonipat realty demand.",
    "Haryana RERA has registered over 800 projects ensuring buyer protection.",
    "Stamp duty in Haryana is 7% for male buyers and 5% for female buyers.",
    "Haryana affordable housing scheme offers properties at fixed rates for EWS and LIG.",
    "Properties within 500 metres of metro stations appreciate 10-15% faster than city average.",
    "New highway or expressway announcements increase nearby property values within 6-12 months.",
    "Airport proximity raises commercial and residential property values significantly.",
    "Smart city project zones carry a 5-10% price premium over comparable non-smart areas.",
    "SEZ and IT park proximity increases residential rental demand substantially.",
    "NRI investment in Indian real estate is fully permitted under FEMA regulations.",
    "RERA registration is mandatory for all real estate projects above 500 sq metres in India.",
    "GST on under-construction properties is 5% without input tax credit benefit.",
    "Capital gains tax on property held over 2 years is 20% with indexation benefits.",
    "Home loan interest deduction up to Rs 2 lakh per year is available under Section 24.",
    "Principal repayment qualifies for deduction up to Rs 1.5 lakh under Section 80C.",
    "TDS at 1% is deducted by the buyer for property transactions above Rs 50 lakhs.",
    "Indian residential real estate grew 10% in 2024 with 4.1 lakh new unit launches.",
    "Luxury housing above Rs 1.5 crore has grown fastest at 18% in 2024.",
    "Affordable housing under Rs 45 lakh saw demand recovery after the COVID-19 slowdown.",
    "Co-living and co-working segments are growing at 25% annually post-pandemic.",
    "Green certified buildings command a 5-7% price premium in major Indian cities.",
    "Commercial office absorption in NCR crossed 10 million sq ft in 2024.",
    "Rental housing demand surged 30% in Bengaluru, Hyderabad and Pune in 2024.",
    "Tier 2 cities like Lucknow, Indore and Surat are emerging real estate hotspots.",
    "Rising home loan interest rates above 8.5% can reduce affordability by 10-15%.",
    "Builder project delays and cancellations remain a key risk for under-construction properties.",
    "Over-supply in certain Noida micro-markets can depress capital appreciation for years.",
    "Legal title disputes are common in older properties and require thorough due diligence.",
    "Flood-prone or seismically active zones carry higher insurance costs and investment risk.",
]

@st.cache_resource
def get_vectorstore():
    try:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(MARKET_CORPUS, emb)
    except Exception as e:
        logger.error(f"Vectorstore init failed: {e}"); return None

def retrieve_market(query: str, vs) -> List[str]:
    if vs is None: return ["Market data unavailable."]
    try:
        return [d.page_content for d in vs.similarity_search(query, k=4)]
    except Exception as e:
        return [f"Retrieval error: {e}"]
