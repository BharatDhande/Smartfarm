import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Generating synthetic data
data = {
    'N': np.random.randint(0, 100, 1000),  # Increased sample size
    'P': np.random.randint(0, 100, 1000),
    'K': np.random.randint(0, 100, 1000),
    'Temperature': np.random.uniform(10, 40, 1000),
    'Humidity': np.random.uniform(20, 90, 1000),
    'PH': np.random.uniform(4, 9, 1000),
    'Rainfall': np.random.uniform(0, 300, 1000),
    'State': np.random.randint(0, 5, 1000),
    'City': np.random.randint(0, 5, 1000),
    'Suitable Crop': np.random.choice(['Wheat', 'Rice', 'Maize', 'Sugarcane', 'Cotton', 'Jute', 'Coffee', 'Tea', 'Barley', 'Soybean'], 1000)
}

df = pd.DataFrame(data)

# Encode the target column
df['Suitable Crop'] = df['Suitable Crop'].astype('category').cat.codes

# Define features and labels
X = df.drop(['Suitable Crop'], axis=1)
y = df['Suitable Crop']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(best_model, 'crop_recommendation_model.pkl')