# PriceLens

*~ A smart ML-powered tool to predict car prices ~*

---

## 📝 What does this app do?

This app helps you estimate the **resale value** of used cars in India using advanced machine learning algorithms! Just enter your car details and get an instant price prediction.

---

## 🔧 **Dashboard Inputs** *(What you need to enter)*

### **Car Specifications:**
- **Brand** ➜ Maruti, Hyundai, Honda, Toyota, Ford, Mahindra, BMW, etc.
- **Vehicle Age** ➜ How old is your car? (0-25 years)
- **Kilometers Driven** ➜ Total distance covered (0-500,000 km)
- **Seller Type** ➜ Dealer / Individual / TrustMark
- **Fuel Type** ➜ Petrol, Diesel, CNG, LPG, Electric
- **Transmission** ➜ Manual or Automatic
- **Mileage** ➜ Fuel efficiency (5-50 kmpl)
- **Engine** ➜ Engine capacity (600-6000 CC)
- **Max Power** ➜ Power output (30-600 bhp)
- **Seats** ➜ Number of seats (2, 4, 5, 6, 7, 8)

---

## 📊 **App Output** *(What you get)*

### **Primary Output:**
- **Estimated Price** ➜ ₹X,XX,XXX.XX (in Indian Rupees)

### **Market Analytics Dashboard:**
- 📈 **Average Resale Price by Brand** (Bar Chart)
- 📉 **Price vs Vehicle Age Trend** (Line Chart) 
- 📦 **Price Distribution by Fuel Type** (Box Plot)
- 🏷️ **Market KPIs**: Median Price, Median Age, Total Listings

---

## 🎯 **How it works**

```
Input Car Details → ML Model Processing → Price Prediction
```

1. **Enter** your car specifications in the sidebar
2. **Click** "Predict Price" button
3. **Get** instant price estimation with market insights!

---

## 🔬 **Technology Stack**

- **Frontend**: Streamlit Dashboard
- **Backend**: Flask API
- **ML Pipeline**: Feature Engineering + Predictive Models
- **Visualization**: Altair Charts
- **Data Processing**: Pandas, NumPy

---

## 🚀 **Quick Start**

```bash
# Run the dashboard
streamlit run streamlit_dashboard.py

# Start backend server  
python app.py
```

---

## 📈 **Key Features**

✅ **Real-time price prediction**  
✅ **Interactive market analytics**  
✅ **User-friendly interface**  
✅ **Comprehensive car feature support**  
✅ **Market trend visualizations**  


