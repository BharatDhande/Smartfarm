import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern

# Path to dataset
base_dir = os.getcwd()
dataset_path = os.path.join(base_dir, "home", "static", "disease_detection")
model_save_path = os.path.join(base_dir, "home", "models", "disease_model.pkl")

# Ensure dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"🚨 Dataset folder not found: {dataset_path}")

# Feature extraction function
def extract_features(image):
    # Resize image for consistency
    image = cv2.resize(image, (128, 128))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract color histogram features
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Extract texture features using Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)

    return np.hstack([hist, hist_lbp])

# Load dataset and labels
features = []
labels = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                features.append(extract_features(image))
                labels.append(category)  # <-- Keeping the full label name like 'tomato_spot'

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and label encoder
with open(model_save_path, "wb") as f:
    pickle.dump((clf, label_encoder), f)

print("✅ Disease Detection Model Trained & Saved Successfully!")
