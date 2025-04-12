# Reinforcement Learning-Based Intrusion Detection and Preemptive Defense System with Context-Aware Threat Analysis

## Overview
This project implements a **Reinforcement Learning-Based Intrusion Detection and Preemptive Defense System** that uses Deep Q-Networks (DQN) for threat detection and proactive defense strategies. The system is designed to detect network intrusions and take preemptive defense actions based on context-aware analysis. The model is trained using the **NSL-KDD** dataset and can be deployed for real-time packet classification.

The system leverages **Reinforcement Learning (RL)** for dynamic decision-making and **Scapy** for capturing and analyzing network packets in real-time.

## Key Features
- **Reinforcement Learning (DQN):** The model uses Deep Q-Networks (DQN) to learn the optimal policy for detecting network intrusions and triggering defense actions.
- **Intrusion Detection:** Detects various types of intrusions in network traffic, such as DoS, R2L, U2R, and Probe.
- **Real-Time Packet Classification with Scapy:** The system uses **Scapy** to capture live network packets and classify them as **ALLOW** or **BLOCK** based on the learned model.
- **Context-Aware Features:** Extracts relevant features from network packets (e.g., source IP, destination IP, time of access) to make classification decisions.
- **Model Training and Evaluation:** Includes training of the DQN model, performance evaluation with confusion matrices, accuracy, and ROC curves.
- **Streamlit Dashboard:** A user-friendly **Streamlit** dashboard for visualizing training progress, evaluating model performance, and monitoring real-time packet classification.

## Project Structure
- `data/` : Contains the **NSL-KDD** dataset and any preprocessed data.
- `models/` : Code for training the DQN model and related files.
- `streamlit_dashboard/` : Streamlit-based dashboard for real-time visualizations and monitoring.
- `src/` : Core source code for RL algorithms, model training, packet capture, and evaluation.
- `logs/` : Logs for real-time packet classification decisions.
- `requirements.txt` : Required Python libraries and dependencies for the project.
- `README.md` : Project documentation and instructions.

1. **Clone the repository**

bash
git clone https://github.com/your-username/rl-ids.git
cd rl-ids
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run rl_ids_app.py
📊 Dataset
The project uses the NSL-KDD dataset, an improved version of the original KDD Cup 1999 dataset.

Download from: https://www.unb.ca/cic/datasets/nsl.html

Make sure the files KDDTrain+.csv and KDDTest+.csv are inside the data/ folder.

⚙️ Environment
Python 3.9+

Streamlit

Pandas

NumPy

Scikit-learn

TensorFlow / PyTorch (as per RL implementation)

🧠 Author
Yukesh Kumar
B.Tech CSE (AI & ML), LPU


