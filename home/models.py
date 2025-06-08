# home/models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from django.db import models

class ContactUs(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.name} ({self.email})"

def build_and_train_crop_model():
    # Generate synthetic dataset for crop recommendation
    data = pd.DataFrame({
        'N': np.random.randint(0, 200, 1500),
        'P': np.random.randint(0, 200, 1500),
        'K': np.random.randint(0, 200, 1500),
        'temperature': np.random.uniform(10, 40, 1500),
        'humidity': np.random.uniform(10, 100, 1500),
        'ph': np.random.uniform(4, 10, 1500),
        'rainfall': np.random.uniform(0, 300, 1500),
        'label': np.random.choice(['rice', 'wheat', 'maize', 'sugarcane', 'cotton', 'millets', 'barley', 
                                   'coconut', 'banana', 'coffee', 'orange', 'tomato', 'potato', 
                                   'soybean', 'groundnut'], 1500)
    })
    
    # Preparing data
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Saving the model
    model_path = os.path.join('home', 'models', 'crop_recommendation_model.pkl')
    joblib.dump(model, model_path)