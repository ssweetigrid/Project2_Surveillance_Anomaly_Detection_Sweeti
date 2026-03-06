"""
Surveillance Anomaly Detection — Streamlit Demo
Interactive anomaly timeline scrubber with sequence viewer.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import math
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Surveillance Anomaly Detection", layout="wide")
st.title("🚨 Surveillance Anomaly Detection")
st.markdown("Real-time anomaly detection in traffic camera footage using Transformer Autoencoders")

# ============================
# Model Definitions (must match training)
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=2048, model_dim=512, num_heads=8,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=1024,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        return self.decoder(encoded)


# ============================
# Sidebar Configuration
# ============================
st.sidebar.header("⚙️ Configuration")
window_size = st.sidebar.selectbox("Window Size", [8, 16], index=0)
stride = st.sidebar.selectbox("Stride", [3, 5], index=1)
threshold_sigma = st.sidebar.slider("Threshold (σ)", 1.5, 4.0, 2.5, 0.1)
scoring = st.sidebar.selectbox("Scoring Method", ["max", "max_std", "mean"])


# ============================
# Data Loading (Cached)
# ============================
@st.cache_data
def load_data():
    test_features = np.load("saved_features/test_features_full.npy")
    train_mean = np.load("saved_features/train_mean.npy")
    train_std = np.load("saved_features/train_std.npy")
    test_features = (test_features - train_mean) / train_std

    index_df = pd.read_csv("Test/index_test.csv")
    anomaly_df = pd.read_csv("Test/anomaly-labels.csv")
    anomaly_df = anomaly_df[anomaly_df["label"] != -1]

    # Build aligned labels (sorted shard order matching feature extraction)
    shard_folders = sorted(
        [f for f in os.listdir("Test") if os.path.isdir(os.path.join("Test", f))]
    )
    aligned_rows = []
    for folder in shard_folders:
        shard_tar = folder + ".tar"
        shard_df = index_df[index_df["shard"] == shard_tar].copy()
        shard_df = shard_df.sort_values("filename").reset_index(drop=True)
        aligned_rows.append(shard_df)

    labels_df = pd.concat(aligned_rows, ignore_index=True)
    labels_df["gt"] = 0
    for _, row in anomaly_df.iterrows():
        mask = (
            (labels_df["timestamp_utc_ms"] >= row["start_timestamp"])
            & (labels_df["timestamp_utc_ms"] <= row["end_timestamp"])
        )
        labels_df.loc[mask, "gt"] = 1

    return test_features, labels_df


@st.cache_resource
def load_model(exp_dir):
    model = TransformerAutoencoder()
    model_path = os.path.join(exp_dir, "model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


test_features, labels_df = load_data()

exp_map = {
    8: "experiments/exp2_transformer_w8_s5",
    16: "experiments/exp1_transformer_w16_s5",
}
exp_dir = exp_map.get(window_size, "experiments/exp2_transformer_w8_s5")
model = load_model(exp_dir)


# ============================
# Compute Anomaly Scores (Cached)
# ============================
@st.cache_data
def compute_scores(_model_state, features, ws, s, score_type):
    model_local = TransformerAutoencoder()
    model_local.load_state_dict(_model_state)
    model_local.eval()

    sequences = []
    for i in range(0, len(features) - ws, s):
        sequences.append(features[i : i + ws])
    sequences = np.array(sequences)

    errors = []
    with torch.no_grad():
        for i in range(0, len(sequences), 128):
            batch = torch.tensor(sequences[i : i + 128], dtype=torch.float32)
            output = model_local(batch)
            frame_mse = torch.mean((output - batch) ** 2, dim=2)
            if score_type == "max":
                score = torch.max(frame_mse, dim=1)[0]
            elif score_type == "max_std":
                score = torch.max(frame_mse, dim=1)[0] + torch.std(frame_mse, dim=1)
            else:
                score = torch.mean(frame_mse, dim=1)
            errors.extend(score.numpy())
    return np.array(errors)


model_state = model.state_dict()
errors = compute_scores(model_state, test_features, window_size, stride, scoring)

# Build sequence labels
frame_gt = labels_df["gt"].values
seq_labels = []
for i in range(0, len(frame_gt) - window_size, stride):
    window = frame_gt[i : i + window_size]
    seq_labels.append(1 if np.any(window == 1) else 0)
seq_labels = np.array(seq_labels[: len(errors)])
errors = errors[: len(seq_labels)]

threshold = errors.mean() + threshold_sigma * errors.std()


# ============================
# Metrics
# ============================
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(seq_labels, errors) if len(np.unique(seq_labels)) > 1 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", f"{auc:.4f}")
col2.metric("Total Sequences", f"{len(errors):,}")
col3.metric("Anomaly Sequences", f"{int(seq_labels.sum()):,}")
col4.metric("Threshold", f"{threshold:.4f}")


# ============================
# Anomaly Score Timeline
# ============================
st.subheader("📊 Anomaly Score Timeline")

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(errors, linewidth=0.3, alpha=0.7, color="blue")
ax.axhline(
    threshold, color="red", linestyle="--", linewidth=1,
    label=f"Threshold ({threshold_sigma}σ)",
)
ax.fill_between(
    range(len(seq_labels)), 0, errors.max() * seq_labels,
    alpha=0.15, color="red", label="GT Anomaly",
)
ax.set_xlabel("Sequence Index")
ax.set_ylabel("Reconstruction Error")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)


# ============================
# Sequence Inspector
# ============================
st.subheader("🔍 Sequence Inspector")
seq_idx = st.slider("Select sequence index", 0, len(errors) - 1, 0)
start_frame = seq_idx * stride

label_text = "🔴 Anomaly" if seq_labels[seq_idx] else "🟢 Normal"
st.write(
    f"**Score:** {errors[seq_idx]:.4f} | **GT:** {label_text} "
    f"| **Frames:** {start_frame} – {start_frame + window_size}"
)

cols = st.columns(min(window_size, 8))
for i in range(min(window_size, 8)):
    frame_idx = start_frame + i
    if frame_idx < len(labels_df):
        row = labels_df.iloc[frame_idx]
        shard = row["shard"].replace(".tar", "")
        filename = row["filename"]
        img_path = os.path.join("Test", shard, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            icon = "🔴" if row["gt"] else "🟢"
            cols[i].image(img, caption=icon, use_container_width=True)


# ============================
# Top Detected Anomalies
# ============================
st.subheader("🔥 Top Detected Anomalies")
top_k = st.sidebar.slider("Top-K Anomalies", 5, 50, 10)
top_idx = np.argsort(errors)[-top_k:][::-1]
top_df = pd.DataFrame(
    {
        "Sequence": top_idx,
        "Score": errors[top_idx],
        "GT Label": [
            "Anomaly" if seq_labels[i] else "Normal" for i in top_idx
        ],
        "Start Frame": top_idx * stride,
    }
)
st.dataframe(top_df, use_container_width=True)


# ============================
# Anomaly Label Legend
# ============================
with st.expander("📋 Anomaly Label Legend"):
    anomaly_labels = {
        1: "Change of lane", 2: "Late turn", 3: "Cutting inside turns",
        4: "Driving on centerline", 5: "Moving out of way for emergency vehicle",
        6: "Short wait at intersection", 7: "Long wait at empty intersection",
        8: "Too far on main road while waiting", 9: "Random stopping on road",
        10: "Random slowing down", 11: "Fast reckless driving",
        12: "Slow insecure driving", 13: "Weird movement",
        14: "Moving backwards", 15: "Approaching slow cars unusually",
        16: "Traffic tie-up", 17: "Almost cut off another vehicle",
        18: "Strong cut-off", 19: "Almost collision",
        20: "Driving into oncoming lane", 21: "Illegal turn",
        22: "Short wrong way driving", 23: "Wrong way driver",
        24: "Multiple turns in roundabout", 25: "Broken down vehicle",
        26: "Stopping in middle of street", 27: "Stopping at pedestrian crossing",
        28: "Driving off the road", 29: "Driving on sidewalk",
        30: "Strong sudden braking", 31: "Swerve to avoid vehicle",
        32: "Extreme risky driving",
    }
    for k, v in anomaly_labels.items():
        st.write(f"**{k}:** {v}")
