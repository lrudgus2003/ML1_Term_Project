"""
모델 평가: Accuracy, F1-score, Precision, Recall

수정 내용:
- 누적수익률/샤프비율 제거 (정규화된 log_return으로 인한 계산 오류)
- Accuracy / F1 / Precision / Recall 기준으로 평가
- 결과를 results/model_metrics.csv로 저장

실행법:
    python 06_evaluate.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import os
from config import DATA_DIR, BASE_DIR


RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_model(name, y_true, y_pred):
    """단일 모델 평가 결과를 출력합니다."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Down', 'Up'], zero_division=0)}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    # 예측 결과 로드
    pred_path = os.path.join(DATA_DIR, "predictions.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"predictions.npz 파일이 없습니다: {pred_path}\n"
            "먼저 05_train_models.py를 실행하세요."
        )

    pred_data = np.load(pred_path)
    y_test = pred_data["y_test"]

    print(f"테스트 샘플 수: {len(y_test)}")
    print(f"  Down(0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"  Up(1):   {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

    model_names = ["arima", "lstm", "transformer"]
    results = []

    for name in model_names:
        if name in pred_data:
            r = evaluate_model(name.upper(), y_test, pred_data[name])
            results.append(r)

    # 비교 테이블 출력
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*40}")
        print("  모델 비교")
        print(f"{'='*40}")
        print(df.to_string(index=False))

        # CSV 저장
        save_path = os.path.join(RESULT_DIR, "model_metrics.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
