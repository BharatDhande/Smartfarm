import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Define paths
model_dir = "home/models"
os.makedirs(model_dir, exist_ok=True)
fertilizer_model_path = os.path.join(model_dir, "fertilizer_model.pkl")

# Generate synthetic data
np.random.seed(0)
num_samples = 5000
plants = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Soybean', 'Groundnut', 'Sunflower', 'Tomato', 'Potato']
plant_nutrients = {
    'Wheat': (120, 60, 40),
    'Rice': (100, 50, 30),
    'Maize': (150, 70, 50),
    'Cotton': (180, 60, 60),
    'Sugarcane': (250, 90, 120),
    'Soybean': (20, 40, 60),
    'Groundnut': (30, 70, 90),
    'Sunflower': (60, 90, 90),
    'Tomato': (80, 40, 50),
    'Potato': (150, 50, 100)
}

features = []
labels = []

for plant, (n, p, k) in plant_nutrients.items():
    for _ in range(num_samples // len(plants)):
        nitrogen = np.random.normal(n, 10)
        phosphorus = np.random.normal(p, 5)
        potassium = np.random.normal(k, 5)
        features.append([nitrogen, phosphorus, potassium])
        labels.append(plant)

features = np.array(features)
labels = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🌟 Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open(fertilizer_model_path, 'wb') as f:
    pickle.dump(best_model, f)
