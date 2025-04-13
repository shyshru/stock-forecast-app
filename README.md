# 📈 Stock Forecast App

A full-stack web application that predicts and visualizes future stock prices using an LSTM (Long Short-Term Memory) neural network.

## 🚀 Project Overview

This project leverages:

- 🧠 **Machine Learning (LSTM)** to forecast stock prices
- 🧪 **FastAPI** to serve predictions via REST API
- 🌐 **React + Chart.js + Tailwind CSS** for interactive frontend

---

## 🧱 Tech Stack

### 🔙 Backend (ML API)
- Python
- FastAPI
- TensorFlow / Keras (LSTM Model)
- Scikit-learn
- yFinance (data source)

### 🌐 Frontend (User Interface)
- React
- TypeScript
- Chart.js (for graphing)
- Tailwind CSS (styling)

---

## 📦 Features

- Predict stock prices for the next N days
- Choose any public stock symbol (e.g., AAPL, GOOGL, MSFT)
- Visualize forecasts with interactive charts
- Full-stack integration between ML model and frontend UI

---

## 📁 Project Structure

stock-forecast-app/

├── backend/

│   └── main.py, model_utils.py, venv, etc.

├── frontend/

│   ├── public/

│   ├── src/

│   │   ├── components/

│   │   │   └── Chart.js

│   │   └── App.js

│   ├── tailwind.config.js

│   ├── postcss.config.js

│   └── package.json

