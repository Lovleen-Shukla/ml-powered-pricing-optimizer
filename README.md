# 📊 ML-Powered Dynamic Pricing & Revenue Optimization System

An **end-to-end decision-focused machine learning system** that predicts product demand and recommends **optimal prices to maximize revenue**, deployed with an interactive business dashboard.

🔗 **Live Demo:** *(add your Streamlit Cloud URL here)*  
🔗 **GitHub Repository:** *(this repo)*

---

## 🚀 Project Overview

Most machine learning projects stop at prediction.  
This project goes further by **converting ML predictions into actionable business decisions**.

The system:
1. Forecasts product demand using historical pricing and time-based features  
2. Simulates multiple pricing scenarios  
3. Identifies the **revenue-maximizing price**  
4. Presents insights through a **business-friendly interactive dashboard**

---

## 🧠 Key Features

- 📈 Demand forecasting using **Random Forest**
- 💰 Revenue-based **pricing optimization**
- 🔁 Price simulation within business constraints
- 📊 Interactive **Streamlit dashboard**
- 🧩 Business-friendly controls & explanations
- 🧼 Clean project structure & reproducible pipeline

---

## 📊 Dashboard Highlights

- Select product (StockCode)
- Define lower & upper price bounds
- View:
  - Current price
  - Recommended price
  - Expected demand
  - Expected revenue
- Revenue vs Price visualization for decision justification

---

## 🧪 Data & Modeling

### 🔹 Data Preparation
- Cleaned transactional retail data
- Removed returns & invalid prices
- Aggregated daily product-level demand
- Captured seasonality and pricing patterns

### 🔹 Features Used
- `Price`
- Day of month
- Month
- Day of week
- Weekend indicator

### 🔹 Models
- **Baseline:** Linear Regression  
- **Final Model:** Random Forest Regressor  
  - Chosen for non-linear demand behavior
  - Feature importance confirms **price sensitivity**

---

## 💡 Pricing Optimization Logic

For each product:
1. Generate a range of candidate prices  
2. Predict demand at each price  
3. Compute expected revenue  
Revenue = Price × Predicted Demand

4. Select the price that maximizes revenue  

This demonstrates **decision-focused ML**, commonly used in real-world pricing systems.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib  
- **Modeling:** Random Forest  
- **Dashboard:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure
```bash
dynamic-pricing-ml/
├── dashboard/
│ └── app.py
├── src/
│ ├── init.py
│ ├── features.py
│ ├── optimizer.py
│ └── train_model.py
├── notebooks/
│ ├── eda.ipynb
│ ├── demand_model.ipynb
│ └── pricing_optimization.ipynb
├── data/
│ └── processed/
│ ├── daily_demand.csv
│ └── demand_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/ml-powered-pricing-optimizer.git
cd ml-powered-pricing-optimizer
```
### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the dashboard
```bash
streamlit run dashboard/app.py
```
---
## 🌍 Deployment
-The application is deployed using Streamlit Cloud, enabling real-time interaction with the pricing model via a public URL.

## 🎯 Use Case & Impact

This project demonstrates how machine learning can be used to:
- Support pricing decisions
- Balance demand and revenue trade-offs
- Provide explainable, business-friendly outputs

It is especially relevant for:
- Data Science roles
- Machine Learning Engineer roles
- Analytics & decision-science positions

## 📌 Key Takeaway
-This project showcases how ML models can be transformed into real business decision systems, not just predictive tools.

## 🙌 Author
Lovleen Shukla
(Aspiring Data Scientist / Machine Learning Engineer)
