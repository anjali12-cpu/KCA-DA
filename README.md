# Predictive Model (Linear Regression)

This repository now contains a simple predictive model implementation in Python.

## What it does
- Loads data from a CSV file.
- Splits data into train/test sets.
- Trains a linear regression model (gradient descent, from scratch).
- Reports evaluation metrics (`MSE` and `R²`).
- Optionally generates an SVG visualization (actual vs predicted values).

## Files
- `predictive_model.py` – model training + evaluation script.
- `sample_data.csv` – tiny example dataset.

## Run
```bash
python3 predictive_model.py sample_data.csv --target price
```

## Visualize predictions
```bash
python3 predictive_model.py sample_data.csv --target price --visualize results.svg
```

This writes a simple scatter plot (`results.svg`) comparing actual vs predicted values.

You can replace `sample_data.csv` with your own dataset and set `--target` accordingly.
