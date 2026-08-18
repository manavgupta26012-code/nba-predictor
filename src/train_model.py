"""
train_model.py
---------------
Trains a logistic regression win-probability model on the model-ready
dataset. Uses a TIME-BASED split (train on earlier games, test on later
games) rather than a random split, because random splits leak future
information into training for a time-series problem like this.

Usage:
    python src/train_model.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "DIFF_PTS", "DIFF_FG_PCT", "DIFF_FG3_PCT", "DIFF_FT_PCT",
    "DIFF_REB", "DIFF_OREB", "DIFF_DREB", "DIFF_AST", "DIFF_STL",
    "DIFF_BLK", "DIFF_TOV", "DIFF_PLUS_MINUS", "DIFF_WIN_PCT",
    "DIFF_REST_DAYS", "DIFF_BACK_TO_BACK",
    "H2H_HOME_WIN_PCT", "H2H_AVG_MARGIN",
    "DIFF_MISSING_IMPACT",  # v7: combined Game-Score impact of any missing top-3 players
    # Dropped after testing (didn't improve accuracy/AUC, ranked near-zero
    # importance): DIFF_MOMENTUM_SCORE, DIFF_GAMES_LAST5, DIFF_CLUTCH_WIN_PCT
]
TARGET_COL = "HOME_WIN"

# Confidence buckets for the calibration table: how often has the model
# actually been right when it was THIS confident? Sportsbooks never publish
# this about their own lines -- it's a real trust signal for the dashboard.
CONFIDENCE_BUCKETS = [0.0, 0.10, 0.20, 0.30, 0.50, 1.01]
CONFIDENCE_LABELS = ["Toss-up (0-10%)", "Slight edge (10-20%)", "Moderate (20-30%)",
                      "Confident (30-50%)", "Very confident (50%+)"]


def build_calibration_table(y_test, probs) -> pd.DataFrame:
    """
    Confidence = |prob - 0.5| * 2, on a 0-1 scale.
    For each confidence bucket, reports how often the model's predicted
    winner was actually correct within that bucket.
    """
    confidence = (pd.Series(probs) - 0.5).abs() * 2
    predicted_home_win = (pd.Series(probs).values >= 0.5).astype(int)
    correct = (predicted_home_win == y_test.values).astype(int)

    bucket = pd.cut(confidence, bins=CONFIDENCE_BUCKETS, labels=CONFIDENCE_LABELS, include_lowest=True)
    table = pd.DataFrame({"bucket": bucket, "correct": correct}).groupby("bucket", observed=True).agg(
        n_games=("correct", "size"),
        actual_accuracy=("correct", "mean"),
    ).reset_index()
    return table


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("GAME_DATE")
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def main():
    df = pd.read_csv(DATA_DIR / "model_ready.csv", parse_dates=["GAME_DATE"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    train, test = time_based_split(df)
    print(f"Train: {len(train)} games ({train['GAME_DATE'].min().date()} - {train['GAME_DATE'].max().date()})")
    print(f"Test:  {len(test)} games ({test['GAME_DATE'].min().date()} - {test['GAME_DATE'].max().date()})")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])
    y_train, y_test = train[TARGET_COL], test[TARGET_COL]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Model performance (held-out, later games) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"Log loss: {log_loss(y_test, probs):.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.3f}")

    # Naive "always pick home team" baseline, since home court advantage
    # alone is a strong predictor in the NBA -- your model needs to beat this.
    baseline_acc = accuracy_score(y_test, [1] * len(y_test))
    print(f"Baseline (always predict home win): {baseline_acc:.3f}")

    coef_report = pd.Series(model.coef_[0], index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print("\n--- Feature importance (standardized coefficients) ---")
    print(coef_report.to_string())

    calibration = build_calibration_table(y_test, probs)
    print("\n--- Calibration: accuracy by confidence bucket ---")
    print(calibration.to_string(index=False))
    calibration.to_csv(DATA_DIR / "calibration_table.csv", index=False)
    print(f"Saved calibration table -> {DATA_DIR / 'calibration_table.csv'}")

    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLS},
                MODEL_DIR / "win_probability_model.pkl")
    print(f"\nSaved model -> {MODEL_DIR / 'win_probability_model.pkl'}")


if __name__ == "__main__":
    main()
