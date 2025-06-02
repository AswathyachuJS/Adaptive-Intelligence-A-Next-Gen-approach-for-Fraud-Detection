# 🛡️ Adaptive Intelligence: A Next-Gen Approach for Fraud Detection

This project is a Django-based web application that leverages Machine Learning to detect fraudulent activities in financial transactions. It analyzes both credit card and PaySim datasets to identify suspicious patterns, helping users visualize insights through interactive charts using **Chart.js**.

---

## 🚀 Features

- 🔍 Detects fraud in both **credit card** and **PaySim** datasets
- 📊 Interactive data visualization using **Chart.js**
- 🧠 Machine Learning models for accurate prediction
- 🔧 Admin panel for easy management
- 🖼️ Clean and intuitive UI with real-time fraud feedback

---

## 🧰 Tech Stack

|       Frontend       | Backend|     ML       |  Visualization |
|----------------------|--------|--------------|----------------|
| HTML, CSS, Bootstrap | Django | Scikit-learn |     Chart.js   |

---

## 📁 Project Structure

Fraud_detection_Django/
│
├── fraudapp/ # Core application
│ ├── views.py # Logic & predictions
│ ├── templates/ # HTML files
│ └── static/ # CSS & JS (Chart.js)
│
├── ML_Models/ # Trained ML models
├── db.sqlite3 # Default DB
├── manage.py # Django CLI
└── requirements.txt # Project dependencies

## 🙌 Acknowledgments

- Credit card dataset: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- PaySim dataset: [Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1)
