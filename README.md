# Supply Chain Demand Forecasting with Neural Networks

## Overview
This project demonstrates SKU-level demand forecasting using a dense feedforward
neural network (MLP) and a Random Forest regression baseline. It explores nonlinear
relationships among promotions, weather, shipping delays, calendar effects, SKU,
region, and product category.

**Important:** The included dataset is synthetic and intended for educational
demonstration. The reported metrics do not represent production performance.

## Problem statement
- Build a feedforward neural network with batch normalization and dropout.
- Tune hidden-layer dimensions, learning rate, dropout, and L2 regularization
  using randomized hyperparameter search.
- Compare the selected neural network against a Random Forest baseline.

## Dataset
- 15,000 synthetic SKU/day observations
- 100 SKUs and 5 regions
- Target: `order_volume`
- Features include promotion discount, temperature, rainfall, shipping delay,
  day of week, month, and day of year.

## Data split
- 70% training
- 15% validation
- 15% final test

Categorical variables are one-hot encoded. Input features are standardized using
a scaler fit on training data only. The target is standardized for MLP training.

## Models
### Neural network
Dense feedforward MLP with batch normalization, ReLU activations, dropout,
L2 regularization, Adam optimization, and early stopping.

### Baseline
RandomForestRegressor with 200 trees, maximum depth 20, and minimum leaf size 2.

## Test results from the Colab run

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Tuned Neural Network (MLP) | 11.0941 | 13.8871 | 0.8740 |
| Random Forest Baseline | 20.6021 | 24.8785 | 0.5955 |

Lower MAE/RMSE are better; higher R² is better. Results may differ when the
pipeline is rerun because training is stochastic and library/hardware versions vary.

## Repository contents
- `supply_chain_demand.csv`: synthetic dataset
- `train_pipeline.py`: standalone training and evaluation pipeline
- `P3_Supply_Chain_Demand_Forecasting.ipynb`: Colab notebook, if exported separately
- `randomized_search_results.csv`: hyperparameter search results
- `model_metrics.csv`: validation and test metrics from pipeline execution
- `validation_model_comparison.csv`, `test_model_comparison.csv`: notebook comparison outputs
- `test_predictions.csv`: held-out test predictions
- `hyperparameter_convergence.png`: search results visualization
- `tuned_mlp_loss_curves.png`: training and validation curves
- `models/`: saved Keras model, Random Forest, scaler, and feature names
- `model_config.json`: target scaling and selected configuration

## Run locally
Use Python 3.10–3.12 for best compatibility with TensorFlow; install dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
python train_pipeline.py
```

Place `supply_chain_demand.csv` in the project directory before running.

## Limitations
This is a synthetic demonstration, not a validated business forecasting system.
Real deployment would require historical demand, stockout and inventory data,
promotion calendars, actual weather and logistics data, time-aware evaluation,
and monitoring for model drift.
