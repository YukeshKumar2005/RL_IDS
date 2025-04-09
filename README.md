 🛡️ Reinforcement Learning-Based Intrusion Detection System (RL-IDS)

This project is a Streamlit web application that demonstrates a reinforcement learning-based intrusion detection and preemptive defense system using the NSL-KDD dataset.

---

 📌 Features

- Uses **Deep Q-Learning (DQN)** for learning attack patterns.
- Performs **context-aware threat detection** using custom reward logic.
- Visualizes results in an interactive **Streamlit dashboard**.
- Supports **log management**, **custom uploads**, and **live predictions**.

---

 📂 Project Structure
├── data/ # Contains NSL-KDD dataset files │ ├── KDDTrain+.csv │ └── KDDTest+.csv ├── logs/ # Contains runtime logs (git-kept) │ └── .gitkeep ├── rl_ids_app.py # Main Streamlit application file ├── model/ # (Optional) Pretrained model storage ├── utils/ # Helper functions for preprocessing, RL, etc. ├── requirements.txt # Python dependencies └── README.md # This file
## 🚀 How to Run

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


