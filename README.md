# 🏠 EstateAI — Intelligent Real Estate Price Prediction

**Milestone 1 – ML-Based Property Price Prediction System**

---

## 📖 Overview

EstateAI is a Machine Learning-based property price prediction platform built with **Streamlit**. It analyzes historical real estate listing data and predicts sale prices based on structured features such as location, carpet area, and amenities.

---

## 🎯 Objective

- Build an end-to-end ML pipeline for property price prediction
- Perform data preprocessing & feature engineering
- Train and compare regression models
- Evaluate performance using standard metrics
- Deploy with a user-friendly interactive interface

---

## 🧠 Features Used for Prediction

| Feature Type       | Examples                        |
|--------------------|---------------------------------|
| 📍 Location        | Locality (one-hot encoded)      |
| 📐 Property Size   | Carpet Area (sq ft)             |
| 🛏 Rooms           | Bedrooms, Bathrooms             |
| 🏠 Other Attributes| Property type, Face, Residential|
| 📅 Date            | Year, Month extracted from Date |
| 💰 Valuation       | Estimated Market Value          |

---

## ⚙️ Technical Implementation

### 🔹 1. Data Preprocessing
- Handling missing values (mode/median imputation)
- Encoding categorical variables (`pd.get_dummies`)
- IQR-based outlier removal on `Sale Price`, `Estimated Value`, `carpet_area`
- Date feature extraction (month, day)

### 🔹 2. Machine Learning Models

| Model | Details |
|-------|---------|
| 🌲 **Random Forest Regressor** | Recommended — captures non-linear patterns |
| 📈 **Linear Regression** | Baseline comparison model |

### 🔹 3. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **R² Score** | Measures explained variance |
| **MAE** (Mean Absolute Error) | Average prediction error |
| **RMSE** (Root Mean Squared Error) | Penalizes large errors |

---

## 📊 Dashboard Features

- **Market Insights**: Box plot of Sale Price distribution + Area vs Price scatter plot
- **Correlation Heatmap**: Heatmap of correlations between key numerical features (Sale Price, Estimated Value, Carpet Area, Rooms, Bathrooms, Tax Rate)
- **Model Performance**: R², MAE, and RMSE metric cards
- **Predicted vs Actual Chart**: Scatter plot comparing predictions to ground truth
- **Feature Importance**: Top 10 most influential features (Random Forest only)
- **Price Predictor**: Interactive form to get instant price predictions

---

## 🔄 System Workflow

1. Load Dataset (`V3.csv`)
2. Data Cleaning & Preprocessing
3. Feature Engineering (encoding + date extraction)
4. Outlier Removal (IQR method)
5. Model Training (Random Forest / Linear Regression)
6. Model Evaluation (R², MAE, RMSE)
7. Visualizations & Real-Time Prediction via Streamlit UI

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.


---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install -r requirements.txt
```

Run locally:
```bash
streamlit run real_state_project.py
```

---

## 📊 Milestone 1 Deliverables

- ✔️ Working ML-based price prediction system
- ✔️ Proper preprocessing & feature engineering
- ✔️ Performance evaluation (R², MAE, RMSE)
- ✔️ Predicted vs Actual visualization
- ✔️ Feature importance chart
- ✔️ Publicly deployed application
- ✔️ Clean, modular, and well-documented codebase

---

## 🏁 Conclusion

EstateAI establishes a strong foundation in classical machine learning for real estate price prediction, ensuring robust preprocessing, reliable model evaluation with three metrics (R², MAE, RMSE), and an interactive Streamlit dashboard with feature importance and prediction visualizations.
