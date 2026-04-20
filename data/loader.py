import streamlit as st
import pandas as pd

@st.cache_data
def load_data(path: str = "V3.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error("Dataset 'V3.csv' not found. Place it in the project root.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Locality"]        = df["Locality"].fillna(df["Locality"].mode()[0])
    df["carpet_area"]     = df["carpet_area"].fillna(df["carpet_area"].median())
    df["Estimated Value"] = df["Estimated Value"].fillna(df["Estimated Value"].median())
    df = df.drop_duplicates()
    cat_cols = [c for c in ["Locality","Property","Residential","Face"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    if "Date" in df.columns:
        df["Date"]  = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df = df.drop("Date", axis=1)
    return df
