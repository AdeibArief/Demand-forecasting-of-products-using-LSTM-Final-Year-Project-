---

# 📈 LSTM Demand Forecasting with Streamlit Dashboard

## 🚀 Project Overview

This project implements a **Demand Forecasting System** using a **Long Short-Term Memory (LSTM) neural network** built in **PyTorch**.
It predicts future demand trends based on historical sales/stock data and provides **interactive visualizations** through a **Streamlit dashboard**.

The system is designed to assist businesses in **inventory optimization, supply chain management, and decision-making**.

---

## ⚙️ Features

* ✅ **LSTM-based demand prediction** using PyTorch
* ✅ **Streamlit interactive dashboard** for user-friendly forecasting
* ✅ Upload custom **CSV dataset** with columns (`Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`)
* ✅ **Visualizations**:

  * Demand trend forecasting
  * Demand distribution (histograms, boxplots)
  * Category-wise insights (Low, Medium, High demand)
  * Rolling averages for trend smoothing
* ✅ **Business insights**: Average demand, Max/Min demand, Demand category breakdown

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit dashboard  
├── model.py               # LSTM model implementation in PyTorch  
├── utils.py               # Data preprocessing, feature engineering  
├── requirements.txt       # Dependencies  
├── README.md              # Project documentation  
└── sample_data.csv        # Example dataset for testing  
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/AdeibArief/Demand-forecasting-of-products-using-LSTM-Final-Year-Project.git
cd lstm-demand-forecasting
```

2. **Create virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📊 Example Visualizations

### 🔹 Forecasted Demand Trend

*(Predicted vs Actual Demand)*

<img width="700" height="350" alt="Screenshot 2025-04-02 125645" src="https://github.com/user-attachments/assets/dac52997-8148-4ff2-b6c7-0618ec8fc45e" />

### 🔹 Demand Distribution

*(Histogram of demand levels)*

<img width="700" height="350" alt="Screenshot 2025-04-02 123923" src="https://github.com/user-attachments/assets/d81851fb-8450-4b2e-b3d5-92fb8d5d8648" />

### 🔹 Demand Categories

*(Low / Medium / High demand segmentation)*

<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/c7c85f75-4077-45fd-a9c2-48cdfea7893e" />

---

## 🧠 Model Details

* **Architecture:** LSTM (Recurrent Neural Network)
* **Framework:** PyTorch
* **Training Data:** Historical demand/sales/stock closing price data
* **Evaluation Metrics:**

  * Loss Function (MSE)
  * Mean Absolute Error (MAE)
  * R² Score
  * Mean Absolute Percentage Error (MAPE)

---

## 📌 Future Improvements

* Add **multi-step forecasting** (predict multiple future periods)
* Incorporate **external features** (holidays, promotions, market trends)
* Deploy on **cloud platforms** (AWS, GCP, Azure) for enterprise use
* Optimize LSTM with **hyperparameter tuning**

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo, create a new branch, and submit a pull request.

---

## 👨‍💻 Author

Developed by **\[Adeib Arief, Tahura Siddiqua and Sana Amreen]**
📧 Contact: \[[sadeib3@gmail.com](mailto:sadeib3@gmail.com)]
🔗 LinkedIn: \[Adeib Arief]

---
