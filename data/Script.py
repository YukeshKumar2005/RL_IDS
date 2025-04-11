import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- Load Data ---
df = pd.read_csv("data/KDDTrain+.csv", header=None)

# --- Add Column Names (based on NSL-KDD description) ---
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label', 'difficulty'
]
df.columns = columns

# --- Drop 'difficulty' column ---
df.drop(columns=['difficulty'], inplace=True)

# --- Binary Classification: normal (0) vs attack (1) ---
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# --- Encode categorical columns ---
cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# --- Normalize numerical columns ---
scaler = MinMaxScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# --- Save processed file ---
df.to_csv("processed_nsl_kdd.csv", index=False)
print("✅ Preprocessing complete! Saved as processed_nsl_kdd.csv")
