# Football Match Winner Prediction using Logistic Regression

## Overview

This project predicts the winner of a football match using Logistic Regression. The model uses features like team statistics and match data to predict the outcome (Home Win, Draw, Away Win).

## Requirements

- Python 3.8+
- Pandas
- Scikit-Learn
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset should include:
- Home Team Stats (e.g., goals scored, possession)
- Away Team Stats (e.g., goals conceded, shots on target)
- Team Form (last 5 match results)

## Model Training

1. **Data Preprocessing**: Load and clean data, handle missing values.
2. **Feature Selection**: Extract relevant features.
3. **Training**: Split data into train/test sets and train using Logistic Regression.

Example code:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('football_matches.csv')
features = data[['home_team_goals', 'away_team_goals', 'home_team_possession', 'away_team_possession']]
labels = data['match_result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Model Accuracy: {accuracy * 100:.2f}%')
```

## Usage

To predict new matches:
```python
new_match = [[2, 1, 60, 40]]
prediction = model.predict(new_match)
print("Predicted Outcome:", prediction)
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/sachin3104/Football-Prediction-ML.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script to train and test the model.


---
Happy predicting! ⚽🤖
