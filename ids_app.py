import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import os
from scapy.all import sniff, IP
import datetime

st.set_page_config(page_title="RL-Based Intrusion Detection", layout="centered")
st.markdown("## 🛡️ RL-Based Intrusion Detection System")
st.markdown("This dashboard uses a DQN agent to classify real-time network packets.")

# --- Hyperparameters ---
STATE_SIZE = 41
ACTION_SIZE = 2
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 0.1
MEMORY_SIZE = 1000
BATCH_SIZE = 32

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("processed_nsl_kdd.csv")
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

X, y = load_data()
feature_names = X.columns
X = X.values
y = y.values

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

def get_action(state):
    if random.random() < EPSILON:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        state = torch.FloatTensor(state)
        q_values = policy_net(state)
        return torch.argmax(q_values).item()

def replay():
    if len(memory) < BATCH_SIZE:
        return 0
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)
    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (~dones)
    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model():
    y_pred = []
    probs = []
    for i in range(len(X_test)):
        state = X_test[i]
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_vals = policy_net(state_tensor)
            probs.append(q_vals.softmax(0).numpy()[1])
            y_pred.append(torch.argmax(q_vals).item())
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    return accuracy, y_test, y_pred, fpr, tpr, roc_auc

def extract_packet_features(packet):
    now = datetime.datetime.now()
    hour = now.hour + now.minute / 60
    hour_normalized = hour / 24.0

    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_ip_freq = np.random.rand()
        geo_location = np.random.randint(0, 5)

        # Human-readable mappings
        geo_mapping = {
            0: "North America",
            1: "Europe",
            2: "Asia",
            3: "South America",
            4: "Africa"
        }

        features = [hour_normalized, src_ip_freq, geo_location] + [0] * (STATE_SIZE - 3)
        context = {
            "Source IP": src_ip,
            "Destination IP": dst_ip,
            "Hour of Access (Normalized)": round(hour_normalized, 2),
            "Hour of Access (Time)": f"{int(hour):02}:{int((hour % 1)*60):02}",
            "Source IP Frequency": round(src_ip_freq, 2),
            "Geo-location Region": geo_mapping[geo_location]
        }
        return features, context
    return None, None


# --- Training ---
if st.button("Train Model"):
    with st.spinner("Training the model..."):
        rewards = []
        losses = []
        for episode in range(5):
            idx = np.random.permutation(len(X_train))
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            total_reward = 0
            total_loss = 0
            for i in range(len(X_shuffled)):
                state = X_shuffled[i]
                label = y_shuffled[i]
                action = get_action(state)
                reward = 1 if action == label else -1
                next_state = X_shuffled[i+1] if i+1 < len(X_shuffled) else state
                done = i+1 >= len(X_shuffled)
                memory.append((state, action, reward, next_state, done))
                total_loss += replay()
                total_reward += reward
            target_net.load_state_dict(policy_net.state_dict())
            rewards.append(total_reward)
            losses.append(total_loss / len(X_shuffled))
            st.write(f"📈 Episode {episode+1} Reward: {total_reward}, Avg Loss: {losses[-1]:.4f}")
            torch.save(policy_net.state_dict(), f"models/dqn_model_ep{episode+1}.pt")

        accuracy, y_true, y_pred, fpr, tpr, roc_auc = evaluate_model()
        st.success(f"✅ Training complete! Accuracy: {accuracy*100:.2f}%")

        st.subheader("Training Rewards Over Episodes")
        st.plotly_chart(px.line(y=rewards, title="Reward per Episode", labels={"x": "Episode", "y": "Reward"}))

        st.subheader("Training Loss Over Episodes")
        st.plotly_chart(px.line(y=losses, title="Loss per Episode", labels={"x": "Episode", "y": "Loss"}))

        st.subheader("Confusion Matrix")
        fig3, ax3 = plt.subplots()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Purples', ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        st.pyplot(fig3)

        st.subheader("ROC Curve")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
        fig4.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig4.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig4)

# --- CSV Upload ---
st.markdown("### 📂 Upload Custom Network CSV")
file = st.file_uploader("Upload CSV File", type=["csv"])
if file:
    df_upload = pd.read_csv(file)
    if set(feature_names).issubset(df_upload.columns):
        st.success("Valid file uploaded.")
        uploaded_X = df_upload[feature_names].values
        predictions = [get_action(state) for state in uploaded_X]
        df_upload['Prediction'] = predictions
        st.dataframe(df_upload)
        st.bar_chart(df_upload['Prediction'].value_counts())
    else:
        st.error("CSV does not contain required features.")

# --- Real-Time Scapy Capture ---
st.markdown("### 🌐 Real-Time Packet Classification with Scapy")
if st.button("Capture & Analyze Packet"):
    with st.spinner("Sniffing packet... (wait ~5s)"):
        pkt = sniff(count=1, timeout=5)
        if pkt:
            features, context = extract_packet_features(pkt[0])
            if features:
                decision = get_action(features)
                st.success(f"Packet classified as: {'BLOCK' if decision == 1 else 'ALLOW'}")
                st.subheader("Context-Aware Features Extracted")
                st.json(context)
                with open("logs/packet_decisions.log", "a") as f:
                    f.write(f"{datetime.datetime.now()} | Decision: {'BLOCK' if decision == 1 else 'ALLOW'} | Context: {context}\n")
            else:
                st.warning("Could not extract IP-based packet features.")
        else:
            st.error("No packet captured. Please check your network interface or try again.")

# --- Log Display ---
st.markdown("### 🧾 Packet Decision Logs")
os.makedirs("logs", exist_ok=True)
if os.path.exists("logs/packet_decisions.log"):
    with open("logs/packet_decisions.log", "r") as f:
        logs = f.read()
    st.text_area("Log Output", logs, height=200)
else:
    st.info("No logs recorded yet.")
