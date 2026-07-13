import numpy as np
import yaml
import os

config = yaml.safe_load(open("config.yaml"))
seq_len = config["sequence_length"]

def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

os.makedirs("data/processed/tensors", exist_ok=True)

for split in ["train", "val", "test"]:
    print(f"Creating sequences for {split}...")

    data = np.load(f"data/processed/{split}.npy")

    X, y = create_sequences(data)

    # ConvLSTM format
    X = X[:, :, None, :, :]
    y = y[:, None, :, :]

    np.save(f"data/processed/tensors/{split}_X.npy", X)
    np.save(f"data/processed/tensors/{split}_y.npy", y)

print("Sequences created.")