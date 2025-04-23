
# 📈 Stock Forecast App

A full-stack web application that predicts and visualizes future stock prices using an LSTM (Long Short-Term Memory) neural network.

---

## 🚀 Project Overview

This project leverages:

- 🧠 **Machine Learning (LSTM)** to forecast stock prices  
- 🧪 **FastAPI** to serve predictions via a REST API  
- 🌐 **React + Chart.js + Tailwind CSS** for an interactive frontend  

---

## 🧱 Tech Stack

### 🔙 Backend (ML API)

- Python  
- FastAPI  
- TensorFlow / Keras (LSTM Model)  
- Scikit-learn  
- yFinance (for fetching stock data)  

### 🌐 Frontend (User Interface)

- React  
- TypeScript  
- Chart.js (for graphing)  
- Tailwind CSS (for styling)  

---

## 📦 Features

- Predict stock prices for the next **N** days  
- Input any valid stock symbol (e.g., `AAPL`, `GOOGL`, `MSFT`)  
- Visualize forecasts with interactive charts  
- Full-stack integration between ML model and frontend UI  

---

## 📁 Project Structure

```
stock-forecast-app/
├── backend/
│   ├── main.py
│   ├── model_utils.py
│   └── (venv, requirements.txt, etc.)
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   └── Chart.tsx
│   │   └── App.tsx
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── package.json
```
