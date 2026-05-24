"""
ARIMA, LSTM, Transformer 모델 학습 및 비교

각 모델을 학습하고 예측 결과를 저장합니다.

실행법:
    python 05_train_models.py
"""

import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import DATA_DIR, MODEL_DIR, SEED, WINDOW_SIZE

np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────

def load_dataset():
    data = np.load(os.path.join(DATA_DIR, "dataset.npz"))
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


# ──────────────────────────────────────
# 공통: batch 단위 예측
# ──────────────────────────────────────

def predict_in_batches(model, X_test, device, batch_size=256):
    """
    X_test 전체를 한 번에 GPU에 올리지 않고,
    batch 단위로 나누어 예측합니다.
    """
    model.eval()

    test_ds = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    preds_list = []

    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)

            logits = model(xb)

            preds_batch = (
                torch.sigmoid(logits) > 0.5
            ).cpu().numpy().astype(int)

            preds_list.append(preds_batch)

    preds = np.concatenate(preds_list)

    return preds


# ──────────────────────────────────────
# 1. ARIMA
# ──────────────────────────────────────

def train_arima(X_train, X_test, y_train, y_test):
    """
    개선된 AR baseline:
    - AR(20) 사용
    - threshold는 train log_return의 0.59 quantile 사용
    """

    from statsmodels.tsa.ar_model import AutoReg

    p = 20
    log_return_idx = 0
    threshold_quantile = 0.59

    # train 시계열 구성
    train_series = X_train[:, -1, log_return_idx]

    # q59 threshold
    threshold = float(
        np.quantile(train_series, threshold_quantile)
    )

    try:
        model = AutoReg(
            train_series,
            lags=p,
            old_names=False
        )

        fitted = model.fit()

        intercept = fitted.params[0]
        ar_coefs = fitted.params[1:]

        # 각 test sample의 마지막 p개 log_return
        test_windows = X_test[:, -p:, log_return_idx]

        # 벡터화 AR prediction
        forecasts = intercept + test_windows @ ar_coefs[::-1]

        # threshold 적용
        preds = (forecasts > threshold).astype(int)

        print(
            f"  ARIMA/AR: p={p}, threshold=q{int(threshold_quantile * 100)} "
            f"({threshold:.6f})"
        )
        print(
            f"  Prediction ratio: "
            f"Down={(preds == 0).mean():.4f}, "
            f"Up={(preds == 1).mean():.4f}"
        )
    except Exception as e:
        print(f"ARIMA 학습 실패: {e}")
        preds = np.zeros(len(y_test), dtype=int)

    return preds


# ──────────────────────────────────────
# 2. LSTM
# ──────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def train_lstm(
    X_train,
    X_test,
    y_train,
    y_test,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  사용 디바이스: {device}")
    input_size = X_train.shape[2]

    n_val = max(1, int(len(X_train) * 0.1))
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size * 2, shuffle=False
    )

    model = LSTMClassifier(input_size).to(device)

    pos_w = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  LSTM Epoch {epoch + 1}/{epochs}, "
            f"train={total_loss / len(train_loader):.4f}  val={val_loss:.4f}  lr={current_lr:.1e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  → Early stopping (best val={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    preds = predict_in_batches(model, X_test, device, batch_size=batch_size)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm.pt"))
    print(f"  모델 저장: {os.path.join(MODEL_DIR, 'lstm.pt')}")

    return preds


# ──────────────────────────────────────
# 3. Transformer
# ──────────────────────────────────────

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)


def train_transformer(
    X_train,
    X_test,
    y_train,
    y_test,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  사용 디바이스: {device}")
    input_size = X_train.shape[2]

    n_val = max(1, int(len(X_train) * 0.1))
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size * 2, shuffle=False
    )

    model = TransformerClassifier(input_size).to(device)

    pos_w = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Transformer Epoch {epoch + 1}/{epochs}, "
            f"train={total_loss / len(train_loader):.4f}  val={val_loss:.4f}  lr={current_lr:.1e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  → Early stopping (best val={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    preds = predict_in_batches(model, X_test, device, batch_size=batch_size)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer.pt"))
    print(f"  모델 저장: {os.path.join(MODEL_DIR, 'transformer.pt')}")

    return preds


# ──────────────────────────────────────
# 메인
# ──────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test = load_dataset()

    print(f"데이터 로드: Train={X_train.shape}, Test={X_test.shape}")

    results = {}

    print("\n[1/3] ARIMA 학습...")
    results["arima"] = train_arima(X_train, X_test, y_train, y_test)

    print("\n[2/3] LSTM 학습...")
    results["lstm"] = train_lstm(X_train, X_test, y_train, y_test)

    print("\n[3/3] Transformer 학습...")
    results["transformer"] = train_transformer(
        X_train,
        X_test,
        y_train,
        y_test
    )

    pred_path = os.path.join(DATA_DIR, "predictions.npz")

    np.savez(
        pred_path,
        y_test=y_test,
        **results
    )

    print(f"\n예측 결과 저장: {pred_path}")


if __name__ == "__main__":
    main()
