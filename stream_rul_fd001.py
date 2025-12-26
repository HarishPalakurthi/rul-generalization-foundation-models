# stream_rul_fd001.py
import torch
import pickle
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG / ARTIFACT PATHS
# -----------------------------
MODEL_PATH = "artifacts/rul_lstm_fd001.pth"
SCALER_PATH = "artifacts/scaler_fd001.pkl"
FEATURES_PATH = "artifacts/features_fd001.pkl"
ARCH_PATH = "artifacts/model_arch_fd001.pkl"

ALERT_THRESHOLD = 20  # RUL threshold to trigger maintenance alert

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    feature_cols = pickle.load(f)

with open(ARCH_PATH, "rb") as f:
    meta = pickle.load(f)

# -----------------------------
# DEFINE MODEL
# -----------------------------
class RULLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = RULLSTM(input_dim=meta["input_dim"], hidden_dim=meta["hidden_dim"]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

window = meta["window"]

# -----------------------------
# STREAMING PREDICTOR CLASS
# -----------------------------
class RULStreamPredictor:
    def __init__(self, model, scaler, feature_cols, window, device):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.window = window
        self.device = device
        self.buffers = {}  # per-engine rolling buffer

    def process_cycle(self, engine_id, raw_features):
        """
        raw_features: dict {feature_name: value}
        Returns: predicted RUL (float) or None if not enough history
        """

        # Initialize buffer for new engine
        if engine_id not in self.buffers:
            self.buffers[engine_id] = deque(maxlen=self.window)

        # Extract features in correct order

        x_df = pd.DataFrame([raw_features], columns=self.feature_cols)

        # Normalize using training scaler
        x_scaled = self.scaler.transform(x_df)[0]

        # Add to buffer
        self.buffers[engine_id].append(x_scaled)

        # Not enough history yet
        if len(self.buffers[engine_id]) < self.window:
            return None

        # Prepare tensor
        window_data = np.array(self.buffers[engine_id]).reshape(1, self.window, -1)
        xt = torch.tensor(window_data, dtype=torch.float32).to(self.device)

        # Predict RUL
        with torch.no_grad():
            rul_pred = self.model(xt).cpu().numpy().item()

        return max(rul_pred, 0.0)

# -----------------------------
# MAINTENANCE ALERT FUNCTION
# -----------------------------
def maintenance_alert(rul, current_cycle, engine_id):
    if rul < ALERT_THRESHOLD:
        return f"Engine {engine_id}: ⚠️ MAINTENANCE REQUIRED — Current Cycle: {current_cycle}, Predicted RUL: {rul:.1f}"
    return None


# -----------------------------
# MAIN STREAM SIMULATION FUNCTION
# -----------------------------
def simulate_stream(test_df):
    """
    Simulates cycle-by-cycle streaming predictions.
    test_df: pandas DataFrame containing FD001 test data
    """
    streamer = RULStreamPredictor(model, scaler, feature_cols, window, device)
    predictions = {}

    for engine_id, engine_df in test_df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        predictions[engine_id] = []

        for _, row in engine_df.iterrows():
            raw_features = row[feature_cols].to_dict()
            current_cycle = row["cycle"]
            pred_rul = streamer.process_cycle(engine_id, raw_features)
            predictions[engine_id].append(pred_rul)

            # Optional: print alert if RUL below threshold
            if pred_rul is not None:
                alert = maintenance_alert(pred_rul, current_cycle, engine_id)
                if alert:
                    print(alert)


    return predictions

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import pandas as pd

    # Load FD001 test data
    test_df = pd.read_csv(r'D:\Data Science\RUL prediction\nasa-cmaps\versions\1\CMaps\test_FD001.txt', sep=r"\s+", header=None)
    COLUMNS = (
        ["engine_id", "cycle"] +
        [f"op_{i}" for i in range(1, 4)] +
        [f"s{i}" for i in range(1, 22)]
    )
    test_df.columns = COLUMNS

    # Drop unused sensors (same as training)
    DROP_COLS = ["op_3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    test_df = test_df.drop(columns=DROP_COLS)

    # Simulate streaming predictions
    predictions = simulate_stream(test_df)

    # Print last predicted RUL per engine (NASA-style)
    print("\nLast RUL prediction per engine:")
    for eid in sorted(predictions.keys()):
        print(f"Engine {eid}: {predictions[eid][-1]:.2f}")



    # Choose the engine you want to visualize
    engine_id_to_plot = 31  # change this to any engine ID

    # Extract cycles and predicted RUL
    engine_df = test_df[test_df['engine_id'] == engine_id_to_plot].sort_values('cycle')
    cycles = engine_df['cycle'].values
    rul_pred = predictions[engine_id_to_plot]

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(cycles, rul_pred, marker='o', linestyle='-', color='blue', label='Predicted RUL')
    plt.axhline(ALERT_THRESHOLD, color='red', linestyle='--', label=f'Alert Threshold ({ALERT_THRESHOLD})')
    plt.xlabel("Cycle")
    plt.ylabel("Predicted RUL")
    plt.title(f"Predicted RUL vs Cycle for Engine {engine_id_to_plot}")
    plt.grid(True)
    plt.legend()
    plt.show()

if i drive my car on dirt road with bumps my RUL is 100 days and if i drive on a concrete road te RUL increases to 200 this makes sense
But for the above the conditions are not this much varied, is that what you are saying, thats why RUL should only have downward trend (also for fd001 they said its onlt one condiiton so my theory isn ot applicable i guess)