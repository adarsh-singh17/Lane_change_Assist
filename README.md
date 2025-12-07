# 🚘 Lane Change Assist System

This project is a real-time machine-learning based lane change assistance system designed to help a vehicle decide whether it should stay in the current lane or shift left/right based on surrounding conditions.  
The idea is inspired from modern driver-assist technologies found in advanced vehicles, where safety decisions are supported using live sensor data.

---

## 🌟 Project Purpose
The goal of this system is to predict safe lane-change decisions using machine learning and show real-time results through an interactive dashboard.  
It demonstrates how vehicle sensors and AI together can improve road safety.

---

## 🧠 What The Model Predicts
The model can output three decisions:

| Action | Meaning |
|--------|--------|
| KEEP_LANE | Continue straight safely |
| CHANGE_LEFT | Move to the left lane |
| CHANGE_RIGHT | Move to the right lane |

The system also displays:
- Model confidence percentage
- Real-time probability distribution
- Risk/Safety score
- Historical graph of past decisions

---

## 📥 Inputs Used (Simulated Sensor Values)
| Parameter | Description |
|-----------|------------|
| Speed (km/h) | Current vehicle speed |
| Steering angle | Wheel turning angle |
| Left lane clearance (m) | Distance to left boundary |
| Right lane clearance (m) | Distance to right boundary |
| Road curvature (°) | Road bend direction & intensity |
| Visibility | Weather condition |
| Traffic density | Road traffic level |

---

## 📤 System Output
- Recommended lane decision
- Confidence score
- Dynamic safety score (0-100)
- Real-time visualization of road positioning
- Trend graph showing evaluations over time

---

## 🛠 Technologies Used
| Component | Tools |
|-----------|------|
| Programming | Python |
| ML Model | Random Forest Classifier |
| Dataset Generation | NumPy, Pandas |
| Visualization |Plotly & Matplotlib |
| UI Dashboard | Streamlit |

---

## 📈 Model Training Details
- Training dataset size: 2500 generated samples
- Accuracy: ~94%
- Dataset includes realistic combinations of lane distances, steering angles and road curvature

---

## ▶ Run Instructions
### Install required packages:
```bash
pip install -r requirements.txt
streamlit run app.py