# Surf Conditions Predictor - Zarautz

Machine Learning project to predict wave height for surf conditions in Zarautz, Basque Country.

## Project Overview

This project uses historical oceanographic data to predict wave heights, helping surfers plan their sessions.

### Problem Type
**Regression** - Predicting continuous wave height values (in meters)

### Best Model Performance
- **Random Forest Regressor**
  - MAE: 0.061 meters (~6cm average error)
  - RMSE: 0.104 meters
  - R² Score: 0.965

## Dataset

- **Source**: Open-Meteo Marine API
- **Location**: Zarautz (43.28°N, 2.17°W)
- **Period**: January 1 - December 5, 2024
- **Samples**: 8,160 hourly measurements
- **Features**: 10 features after preprocessing

## Features Used

### Raw Features
- `wave_direction`: Direction of waves
- `wave_period`: Time between waves (seconds)
- `wind_wave_height`: Height of wind-generated waves
- `wind_wave_direction`: Direction of wind waves
- `wind_wave_period`: Period of wind waves

### Engineered Features
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)
- `day_of_year`: Day of year (1-365)
- `wind_ratio`: Ratio of wind waves to total waves

**Note**: `wave_height` is the target variable, NOT used as a feature.

## Project Structure
```
surf-predictor/
├── data/
│   ├── raw/              # Raw CSV data from API
│   └── processed/        # Processed data (if needed)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_collection.py    # API data fetching
│   ├── preprocessing.py      # Feature engineering & preprocessing
│   └── model.py              # ML models
├── train_model.py            # Main training script
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd surf-predictor

# Create conda environment
conda create -n surf-predictor python=3.10
conda activate surf-predictor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Collect Data
```bash
python src/data_collection.py
```

### 2. Train Models
```bash
python train_model.py
```

## Models Implemented

1. **Linear Regression** (Baseline)
   - MAE: 0.223m
   - R²: 0.696

2. **Random Forest Regressor** (Best)
   - MAE: 0.061m
   - R²: 0.965

## Key Learnings

- Importance of avoiding data leakage in ML projects
- Feature engineering significantly improves model performance
- Random Forest outperforms Linear Regression for this non-linear problem
- Temporal features (month, hour) are important predictors

## Future Improvements

- [ ] Add more surf spots (Mundaka, Sopelana)
- [ ] Implement time series forecasting (LSTM)
- [ ] Create web interface for predictions
- [ ] Add feature importance visualization
- [ ] Deploy model as API

## Author

Julio Benavides - Data Science Learning Project

## License

MIT
