"""
주가 피처 + 감성 점수를 합쳐 시계열 입력 데이터셋 생성

수정 내용:
1. news_count를 직접 피처로 사용하지 않음
2. news_count를 sentiment_mean, sentiment_std에 가중치로 반영
3. 종목별 train/test split 이후 train 기준으로 Min-Max 정규화
4. 종목별 시간 순서를 유지한 채 시퀀스 생성
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from config import DATA_DIR, WINDOW_SIZE, TEST_RATIO, SEED


FEATURE_COLS = [
    "log_return", "volume", 
    "sma_5", "sma_20", "sma_60", 
    "rsi", "macd", "macd_signal", "macd_hist", 
    "bb_upper", "bb_lower", "bb_width", 
    "sentiment_mean_weighted", "sentiment_std_weighted",
]


def create_sequences(df: pd.DataFrame, window: int, feature_cols: list):
    """시계열 윈도우 데이터 생성"""
    X, y = [], []

    data = df[feature_cols].values
    target = df["target"].values

    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def add_weighted_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    news_count를 감성 피처에 가중치로 반영합니다.

    news_count가 많을수록 해당 날짜의 sentiment_mean/std 신뢰도가 높다고 보고,
    log1p(news_count)를 곱합니다.

    news_count = 0이면 log1p(0) = 0이므로 감성 피처도 0이 됩니다.
    """
    weight = np.log1p(df["news_count"])

    df["sentiment_mean_weighted"] = df["sentiment_mean"] * weight
    df["sentiment_std_weighted"] = df["sentiment_std"] * weight

    return df


def main():
    np.random.seed(SEED)

    features_path = os.path.join(DATA_DIR, "features.csv")
    sentiment_path = os.path.join(DATA_DIR, "sentiment.csv")

    df_feat = pd.read_csv(features_path, parse_dates=["date"])
    df_feat["code"] = df_feat["code"].astype(str).str.zfill(6)

    df_sent = pd.read_csv(sentiment_path, parse_dates=["date"])
    df_sent.rename(columns={"종목코드": "code"}, inplace=True)
    df_sent["code"] = df_sent["code"].astype(str).str.zfill(6)

    # 주가 피처와 감성 점수를 종목코드·날짜 기준으로 병합
    df = df_feat.merge(
        df_sent,
        on=["code", "date"],
        how="left",
        validate="one_to_one"
    )

    # 뉴스가 없는 날은 감성 정보가 없다는 뜻이므로 0으로 채움
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_std"] = df["sentiment_std"].fillna(0)
    df["news_count"] = df["news_count"].fillna(0)

    # news_count를 직접 피처로 쓰지 않고 감성 피처에 반영
    df = add_weighted_sentiment(df)

    # 모델 입력에 필요한 컬럼 결측 제거
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    print(f"병합 데이터: {len(df)}행, 종목 수: {df['code'].nunique()}")

    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []

    used_codes = 0
    skipped_codes = 0

    for code, group in df.groupby("code"):
        # 각 종목 내부에서 날짜 기준으로 정렬
        group = group.sort_values("date").reset_index(drop=True)

        # 윈도우와 최소 test 샘플을 만들 수 없는 종목은 제외
        if len(group) < WINDOW_SIZE + 2:
            skipped_codes += 1
            continue
        
        # 각 종목별로 과거 구간은 train, 미래 구간은 test로 분할
        split_idx = int(len(group) * (1 - TEST_RATIO))

        train_group = group.iloc[:split_idx].copy()

        # test 첫 샘플 생성을 위해 split 이전 WINDOW_SIZE만큼 context 포함
        test_group = group.iloc[split_idx - WINDOW_SIZE:].copy()

        if len(train_group) < WINDOW_SIZE + 1 or len(test_group) < WINDOW_SIZE + 1:
            skipped_codes += 1
            continue

        # 정규화는 train 기준 fit, test는 transform만
        scaler = MinMaxScaler()
        train_group[FEATURE_COLS] = scaler.fit_transform(train_group[FEATURE_COLS])
        test_group[FEATURE_COLS] = scaler.transform(test_group[FEATURE_COLS])

        X_train_part, y_train_part = create_sequences(
            train_group, WINDOW_SIZE, FEATURE_COLS
        )
        X_test_part, y_test_part = create_sequences(
            test_group, WINDOW_SIZE, FEATURE_COLS
        )

        all_X_train.append(X_train_part)
        all_y_train.append(y_train_part)
        all_X_test.append(X_test_part)
        all_y_test.append(y_test_part)

        used_codes += 1

    if not all_X_train or not all_X_test:
        raise ValueError(
            "No train/test sequences were generated. "
            "Check data size or WINDOW_SIZE."
        )

    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_test = np.concatenate(all_X_test)
    y_test = np.concatenate(all_y_test)

    print(f"사용 종목 수: {used_codes}, 제외 종목 수: {skipped_codes}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    output_path = os.path.join(DATA_DIR, "dataset.npz")

    np.savez(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=np.array(FEATURE_COLS),
    )

    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()
