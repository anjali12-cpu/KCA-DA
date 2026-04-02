#!/usr/bin/env python3
"""Simple predictive modeling script using linear regression from scratch."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class Dataset:
    features: List[List[float]]
    target: List[float]


def load_csv(path: Path, target_column: str) -> Dataset:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row.")
        if target_column not in reader.fieldnames:
            raise ValueError(f"Target column '{target_column}' not found in CSV header.")

        feature_columns = [c for c in reader.fieldnames if c != target_column]
        if not feature_columns:
            raise ValueError("CSV must contain at least one feature column.")

        x: List[List[float]] = []
        y: List[float] = []

        for row in reader:
            x.append([float(row[col]) for col in feature_columns])
            y.append(float(row[target_column]))

    if len(x) < 3:
        raise ValueError("Need at least 3 rows of data for train/test split.")

    return Dataset(features=x, target=y)


def train_test_split(dataset: Dataset, test_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    indices = list(range(len(dataset.target)))
    random.Random(seed).shuffle(indices)

    split = int(len(indices) * (1 - test_ratio))
    if split <= 0 or split >= len(indices):
        raise ValueError("test_ratio yields an empty train or test set.")

    train_idx = indices[:split]
    test_idx = indices[split:]

    train = Dataset([dataset.features[i] for i in train_idx], [dataset.target[i] for i in train_idx])
    test = Dataset([dataset.features[i] for i in test_idx], [dataset.target[i] for i in test_idx])
    return train, test


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.feature_means: List[float] = []
        self.feature_stds: List[float] = []

    def _compute_scaler(self, x: Sequence[Sequence[float]]) -> None:
        n_features = len(x[0])
        self.feature_means = []
        self.feature_stds = []
        for j in range(n_features):
            col = [row[j] for row in x]
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / len(col)
            std = var ** 0.5
            self.feature_means.append(mean)
            self.feature_stds.append(std if std > 0 else 1.0)

    def _transform(self, x: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [(row[j] - self.feature_means[j]) / self.feature_stds[j] for j in range(len(row))]
            for row in x
        ]

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        n_samples = len(x)
        n_features = len(x[0])
        self._compute_scaler(x)
        x_scaled = self._transform(x)

        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

        for _ in range(self.epochs):
            dw = [0.0 for _ in range(n_features)]
            db = 0.0

            for row, target in zip(x_scaled, y):
                prediction = self._predict_row(row)
                error = prediction - target

                for j in range(n_features):
                    dw[j] += error * row[j]
                db += error

            for j in range(n_features):
                self.weights[j] -= self.learning_rate * (2 / n_samples) * dw[j]
            self.bias -= self.learning_rate * (2 / n_samples) * db

    def _predict_row(self, row: Sequence[float]) -> float:
        return sum(w * v for w, v in zip(self.weights, row)) + self.bias

    def predict(self, x: Sequence[Sequence[float]]) -> List[float]:
        x_scaled = self._transform(x)
        return [self._predict_row(row) for row in x_scaled]


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1 - (ss_res / ss_tot if ss_tot else 0.0)


def write_prediction_svg(y_true: Sequence[float], y_pred: Sequence[float], out_path: Path) -> None:
    """Create a simple SVG scatter plot of actual vs predicted values."""
    if not y_true or not y_pred:
        raise ValueError("Cannot visualize empty prediction data.")

    width, height = 760, 520
    margin = 60
    all_values = list(y_true) + list(y_pred)
    min_val = min(all_values)
    max_val = max(all_values)
    span = max(max_val - min_val, 1e-9)

    def scale_x(v: float) -> float:
        return margin + ((v - min_val) / span) * (width - 2 * margin)

    def scale_y(v: float) -> float:
        return height - margin - ((v - min_val) / span) * (height - 2 * margin)

    points = [
        f'<circle cx="{scale_x(actual):.2f}" cy="{scale_y(pred):.2f}" r="5" fill="#2563eb" opacity="0.85" />'
        for actual, pred in zip(y_true, y_pred)
    ]

    # Ideal line: y = x
    x1 = scale_x(min_val)
    y1 = scale_y(min_val)
    x2 = scale_x(max_val)
    y2 = scale_y(max_val)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2:.0f}" y="30" text-anchor="middle" font-size="22" font-family="Arial" fill="#111827">
  Actual vs Predicted
</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#374151" stroke-width="2"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#374151" stroke-width="2"/>
<text x="{width / 2:.0f}" y="{height - 15}" text-anchor="middle" font-size="14" font-family="Arial" fill="#111827">Actual</text>
<text x="20" y="{height / 2:.0f}" transform="rotate(-90, 20, {height / 2:.0f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#111827">Predicted</text>
<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="8 6"/>
{''.join(points)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple linear regression predictive model.")
    parser.add_argument("csv_path", type=Path, help="Path to CSV training data")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize",
        type=Path,
        default=None,
        help="Optional path for SVG plot (actual vs predicted), e.g. results.svg",
    )
    args = parser.parse_args()

    dataset = load_csv(args.csv_path, args.target)
    train, test = train_test_split(dataset, args.test_ratio, args.seed)

    model = LinearRegressionGD(args.learning_rate, args.epochs)
    model.fit(train.features, train.target)

    predictions = model.predict(test.features)
    print(f"Test MSE: {mse(test.target, predictions):.4f}")
    print(f"Test R2:  {r2_score(test.target, predictions):.4f}")
    print(f"Weights:  {model.weights}")
    print(f"Bias:     {model.bias:.4f}")
    if args.visualize is not None:
        write_prediction_svg(test.target, predictions, args.visualize)
        print(f"Visualization written to: {args.visualize}")


if __name__ == "__main__":
    main()
