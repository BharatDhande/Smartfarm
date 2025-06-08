# home/utils.py

import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "home/chatbot_model.pkl"
VECTORIZER_PATH = "home/vectorizer.pkl"

TRAINING_DATA = [
    ("What crop should I grow in sandy soil?", "crop_recommendation"),
    ("How do I treat leaf rust in wheat?", "disease_detection"),
    ("What fertilizer is good for rice?", "fertilizer_recommendation"),
    ("Suggest a good crop for clay soil", "crop_recommendation"),
    ("How to cure rice blast disease?", "disease_detection"),
    ("Best fertilizer for corn?", "fertilizer_recommendation")
]

INTENT_RESPONSES = {
    "crop_recommendation": "Based on your query, I recommend crops like maize, rice, or peanuts. Want more guidance?",
    "disease_detection": "It looks like you're asking about disease treatment. Proper fungicides or pesticides can help. Need specifics?",
    "fertilizer_recommendation": "Fertilizers like urea and NPK blends work well. Would you like me to suggest precise dosages?"
}

def train_and_save_model():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([item[0] for item in TRAINING_DATA])
    y = [item[1] for item in TRAINING_DATA]

    model = MultinomialNB()
    model.fit(X, y)

    # Save the model and vectorizer to disk
    with open(MODEL_PATH, 'wb') as model_file, open(VECTORIZER_PATH, 'wb') as vectorizer_file:
        pickle.dump(model, model_file)
        pickle.dump(vectorizer, vectorizer_file)

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as model_file, open(VECTORIZER_PATH, 'rb') as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        train_and_save_model()  # Train and save if not found
        return load_model()  # Load the newly trained model

model, vectorizer = load_model()

def predict_intent(user_input):
    try:
        X_test = vectorizer.transform([user_input])
        intent = model.predict(X_test)[0]
        response = INTENT_RESPONSES.get(intent, "Sorry, I couldn't understand that. Can you rephrase?")
        return response
    except Exception as e:
        return f"Error: {str(e)}"
