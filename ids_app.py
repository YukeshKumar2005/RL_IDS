# Reinforcement Learning-Based Intrusion Detection System using DQN and Streamlit
import pandas as pd
import numpy as np
import random
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from collections import deque

# === 1. Load NSL-KDD Dataset ===
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"
]
df = pd.read_csv("KDDTrain+.csv", names=columns)
df.drop(['difficulty_level'], axis=1, inplace=True)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df[col] = le.fit_transform(df[col])

# === 2. Context-Aware Feature Engineering ===
def add_context_features(data):
    data['hour'] = datetime.datetime.now().hour / 24
    data['src_ip_freq'] = np.random.rand(len(data))
    data['geo_location'] = np.random.randint(0, 5, len(data))
    return data

df = add_context_features(df)
X = df.drop('label', axis=1)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 3. DQN Model ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# === 4. Training Setup ===
input_dim = X_train.shape[1]
output_dim = 2
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
loss_fn = nn.MSELoss()
epsilon = 0.1
gamma = 0.9
batch_size = 64
memory = deque(maxlen=10000)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def get_action(state):
    if random.random() < epsilon:
        return random.randint(0, 1)
    with torch.no_grad():
        return torch.argmax(policy_net(torch.tensor(state, dtype=torch.float32))).item()


def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        state_t = torch.tensor(state, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, dtype=torch.float32)
        target = reward
        if not done:
            target += gamma * torch.max(target_net(next_state_t)).item()
        target_f = policy_net(state_t)
        target_val = target_f.clone().detach()
        target_val[action] = target
        loss = loss_fn(target_f, target_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === 5. Training ===
rewards = []
with open("logs/training_log.txt", "w") as f_log:
    for episode in range(10):
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train.values[idx]
        total_reward = 0
        for i in range(len(X_shuffled)):
            state = X_shuffled[i]
            label = y_shuffled[i]
            action = get_action(state)
            reward = 1 if action == label else -1
            next_state = X_shuffled[i+1] if i+1 < len(X_shuffled) else state
            done = i+1 >= len(X_shuffled)
            memory.append((state, action, reward, next_state, done))
            replay()
            total_reward += reward
        target_net.load_state_dict(policy_net.state_dict())
        rewards.append(total_reward)
        f_log.write(f"Episode {episode+1}, Total Reward: {total_reward}\n")
        torch.save(policy_net.state_dict(), f"models/dqn_model_ep{episode+1}.pt")

# === 6. Evaluation ===
def evaluate_model():
    y_pred = []
    y_true = []
    for i in range(len(X_test)):
        state = X_test[i]
        label = y_test.values[i]
        action = get_action(state)
        y_pred.append(action)
        y_true.append(label)

    acc = sum([1 for i in range(len(y_true)) if y_pred[i] == y_true[i]]) / len(y_true)

    with open("logs/evaluation_metrics.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(str(classification_report(y_true, y_pred)))

    return acc, y_true, y_pred

accuracy, y_true, y_pred = evaluate_model()

# === 7. Streamlit Dashboard ===
st.title("\U0001F6E1️ RL-Based Intrusion Detection System")
st.write("This dashboard uses a DQN agent to classify real-time network packets.")

if st.button("Run Real-Time Detection"):
    with open("logs/realtime_decisions.log", "w") as f_log:
        for _ in range(10):
            sample = np.random.rand(input_dim)
            scaled = scaler.transform([sample])[0]
            prediction = get_action(scaled)
            decision = "BLOCK" if prediction == 1 else "ALLOW"
            st.write(f"\u27A1\uFE0F Incoming Packet: **{decision}**")
            f_log.write(f"{datetime.datetime.now()}: {decision}\n")
            time.sleep(0.5)

st.subheader("Training Rewards Over Episodes")
st.line_chart(rewards)

st.success(f"Final DQN Accuracy on Test Set: {accuracy*100:.2f}%")

# === 8. Confusion Matrix Visualization ===
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# === 9. Real-Time Log Viewer ===
st.subheader("Real-Time Logs")
if os.path.exists("logs/realtime_decisions.log"):
    with open("logs/realtime_decisions.log", "r") as f:
        logs = f.read()
    st.text_area("Log Output", logs, height=200)

# === 10. Custom File Upload for Testing ===
st.subheader("Upload Custom Network CSV")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df_custom = pd.read_csv(uploaded_file)
    st.write("First 5 rows of your data:")
    st.write(df_custom.head())
    # Note: You can extend this to run predictions as well
