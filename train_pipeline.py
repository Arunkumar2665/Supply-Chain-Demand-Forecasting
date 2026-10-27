from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

df = pd.read_csv(PROJECT_DIR / "supply_chain_demand.csv")
target_column = "order_volume"
categorical_features = ["sku_id", "region", "sku_category"]
numeric_features = [
    "promotion_discount", "temperature_c", "rainfall_mm",
    "shipping_delay_days", "day_of_week", "month", "day_of_year"
]

X_raw = df[categorical_features + numeric_features].copy()
y = df[target_column].astype(float)
X = pd.get_dummies(
    X_raw, columns=categorical_features, dtype=float
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

target_mean = float(y_train.mean())
target_std = float(y_train.std())
y_train_s = (y_train.to_numpy() - target_mean) / target_std
y_val_s = (y_val.to_numpy() - target_mean) / target_std

def build_mlp(input_dim, hidden_units, learning_rate,
              dropout_rate, l2_strength):
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for units in hidden_units:
        model.add(layers.Dense(
            units, use_bias=False,
            kernel_regularizer=regularizers.l2(l2_strength)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mae"]
    )
    return model

search_space = {
    "hidden_units": [(64, 32), (128, 64), (128, 64, 32), (256, 128)],
    "learning_rate": [0.0003, 0.001, 0.003],
    "dropout_rate": [0.10, 0.20, 0.30],
    "l2_strength": [0.00001, 0.0001, 0.001],
    "batch_size": [64, 128]
}
candidates = list(ParameterSampler(
    search_space, n_iter=6, random_state=RANDOM_STATE
))

search_rows = []
search_histories = []
for trial, params in enumerate(candidates, start=1):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE + trial)
    model = build_mlp(
        X_train_s.shape[1], params["hidden_units"],
        params["learning_rate"], params["dropout_rate"],
        params["l2_strength"]
    )
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=35, batch_size=params["batch_size"],
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )],
        verbose=0
    )
    best_loss = float(min(history.history["val_loss"]))
    search_rows.append({
        "trial": trial,
        "hidden_units": str(params["hidden_units"]),
        "learning_rate": params["learning_rate"],
        "dropout_rate": params["dropout_rate"],
        "l2_strength": params["l2_strength"],
        "batch_size": params["batch_size"],
        "best_val_loss": best_loss
    })
    search_histories.append((params, history.history))

search_df = pd.DataFrame(search_rows).sort_values("best_val_loss")
search_df.to_csv(PROJECT_DIR / "randomized_search_results.csv", index=False)
best_trial = int(search_df.iloc[0]["trial"])
best_params = candidates[best_trial - 1]

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(RANDOM_STATE)
mlp = build_mlp(
    X_train_s.shape[1], best_params["hidden_units"],
    best_params["learning_rate"], best_params["dropout_rate"],
    best_params["l2_strength"]
)
history = mlp.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=60, batch_size=best_params["batch_size"],
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )],
    verbose=0
)

rf = RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)

def predict_mlp(X_scaled):
    return np.maximum(
        0, mlp.predict(X_scaled, verbose=0).ravel() * target_std + target_mean
    )

def metrics(y_true, y_pred, model_name, split):
    return {
        "split": split, "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

rows = []
for split, true, mlp_pred, rf_pred in [
    ("validation", y_val, predict_mlp(X_val_s), np.maximum(0, rf.predict(X_val))),
    ("test", y_test, predict_mlp(X_test_s), np.maximum(0, rf.predict(X_test)))
]:
    rows.append(metrics(true, mlp_pred, "Tuned Neural Network (MLP)", split))
    rows.append(metrics(true, rf_pred, "Random Forest Baseline", split))

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(PROJECT_DIR / "model_metrics.csv", index=False)
print(metrics_df.round(4).to_string(index=False))

mlp.save(MODEL_DIR / "tuned_demand_mlp.keras")
joblib.dump(rf, MODEL_DIR / "random_forest_baseline.joblib")
joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(list(X.columns), f, indent=2)

config = {
    "random_state": RANDOM_STATE,
    "target_column": target_column,
    "target_mean": target_mean,
    "target_std": target_std,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "selected_mlp_hyperparameters": {
        **best_params, "hidden_units": list(best_params["hidden_units"])
    },
    "dataset_type": "Synthetic demonstration data"
}
with open(PROJECT_DIR / "model_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Search convergence
plt.figure(figsize=(9, 5))
plt.plot(search_df["trial"], search_df["best_val_loss"], marker="o")
plt.xlabel("Randomized Search Trial")
plt.ylabel("Best Validation MSE (scaled target)")
plt.title("Hyperparameter Search Convergence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "hyperparameter_convergence.png", dpi=150)
plt.close()

# Final MLP learning curves
hist_df = pd.DataFrame(history.history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_df["loss"], label="Training")
axes[0].plot(hist_df["val_loss"], label="Validation")
axes[0].set_title("MLP Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(hist_df["mae"], label="Training")
axes[1].plot(hist_df["val_mae"], label="Validation")
axes[1].set_title("MLP MAE (scaled target)")
axes[1].set_xlabel("Epoch")
axes[1].legend()
fig.tight_layout()
fig.savefig(PROJECT_DIR / "tuned_mlp_loss_curves.png", dpi=150)
plt.close(fig)

print("\\nPipeline complete. Artifacts saved in:", PROJECT_DIR)
