#  Real Estate Investment Advisor

##  Project Overview
This project is an end-to-end Machine Learning application that helps evaluate real estate investments by predicting:

- ✔ Whether a property is a **Good Investment (Classification)**
- ✔ Expected **Future Price after 5 years (Regression)**

It includes a full ML pipeline with feature engineering, model training, evaluation, and a deployed interactive web app using Streamlit.

---

## Features
- End-to-end ML pipeline using Scikit-learn
- Feature engineering for real estate data
- Classification model for investment decision
- Regression model for future price prediction
- Interactive Streamlit dashboard
- Model tracking using MLflow
- Clean modular project structure

---

## Dataset Features
The model uses property and locality-based features:

- City, State, Locality
- Property Type, BHK
- Size (SqFt)
- Price (Lakhs)
- Year Built
- Nearby Schools & Hospitals
- Public Transport Accessibility
- Amenities & Infrastructure
- Ownership and Property Status

---

##  Machine Learning Models

### 🔹 Classification Model
- Random Forest Classifier  
- Output: Good Investment (Yes/No)

### 🔹 Regression Model
- Random Forest Regressor  
- Output: Predicted Future Price

---

##  Project Structure

```

real_estate_investment_advisor/
│
├── app_streamlit.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── classifier.pkl
│   └── regressor.pkl
│
├── src/
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   └── eda.ipynb
│
└── data/

````

---

## ⚙ How to Run Locally

### 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/real-estate-investment-advisor.git
cd real-estate-investment-advisor
````

---

### 2. Create virtual environment

```bash
python -m venv venv
```

---

### 3. Activate environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Run Streamlit app

```bash
streamlit run app_streamlit.py
```

---

## Deployment

The app is deployed using **Streamlit Community Cloud**.

 [https://share.streamlit.io](https://share.streamlit.io)

---

## Key Insights

* Property price increases with size and location quality
* Connectivity and infrastructure strongly impact investment value
* Higher BHK generally indicates higher price but with variability
* Location plays a stronger role than physical property features

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Plotly, Seaborn
* Streamlit
* MLflow

---

## Author

Khyathi

Machine Learning & Data Science Enthusiast

---

## Future Improvements

* Add geospatial mapping (real estate heatmaps)
* Integrate live property APIs
* Add deep learning models
* Deploy with Docker + CI/CD pipeline

```

