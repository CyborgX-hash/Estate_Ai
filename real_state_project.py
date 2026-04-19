import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import TypedDict, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langgraph.graph import StateGraph, END