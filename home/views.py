import json
from django.shortcuts import render,redirect
import pickle
import numpy as np
import pandas as pd
import os
import cv2
from .train_disease_model import extract_features
import joblib
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from .forms import ContactUsForm
from django.http import HttpResponse
from .models import build_and_train_crop_model
# import openai
from django.http import JsonResponse
from .chatbot import chatbot_view

# from django.views.decorators.csrf import csrf_exempt
# from .utils import predict_intent
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# # from .train_chatbot import get_response


# Define the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct model directory
model_dir = os.path.join(BASE_DIR, "home", "models")

# Define model paths
crop_model_path = os.path.join(model_dir, "crop_models.pkl")
fertilizer_model_path = os.path.join(model_dir, "fertilizer_model.pkl")
disease_model_path = os.path.join(model_dir, "disease_model.pkl")
state_encoder_path = os.path.join(BASE_DIR, 'home', 'state_encoder.pkl')
city_encoder_path = os.path.join(BASE_DIR, 'home', 'city_encoder.pkl')

# Fertilizer recommendation dictionary
fertilizer_info = {
    "Urea": "Urea is a nitrogen-rich fertilizer used to enhance plant growth and increase crop yield.",
    "DAP": "DAP (Diammonium Phosphate) provides nitrogen and phosphorus, promoting root development.",
    "MOP": "MOP (Muriate of Potash) provides potassium, essential for disease resistance and water regulation.",
    "14-35-14": "A balanced fertilizer suitable for general crop growth, providing essential N-P-K nutrients.",
    "28-28": "Specially formulated to provide equal nitrogen and phosphorus, promoting healthy growth.",
    "17-17-17": "A balanced fertilizer that supports overall plant health and productivity.",
    "20-20": "An effective fertilizer providing equal amounts of nitrogen and phosphorus.",
    "10-26-26": "Formulated to provide phosphorus and potassium for fruit and seed formation.",
    "19-19-19": "Balanced fertilizer providing nitrogen, phosphorus, and potassium for overall plant growth."
}

# Function to safely load models
def load_model(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"🚨 Model file missing or empty: {file_path}")
        return None

# Load trained models
crop_model = load_model(crop_model_path)
fertilizer_model = load_model(fertilizer_model_path)
disease_model = load_model(disease_model_path)
state_encoder = load_model(state_encoder_path)
city_encoder = load_model(city_encoder_path)


def train_crop_model(request):
    build_and_train_crop_model()
    return render(request, 'home/train_success.html')

def crop_recommend(request):
    if request.method == 'POST':
        nitrogen = float(request.POST['nitrogen'])
        phosphorus = float(request.POST['phosphorus'])
        potassium = float(request.POST['potassium'])
        temperature = float(request.POST['temperature'])
        humidity = float(request.POST['humidity'])
        ph = float(request.POST['ph'])
        rainfall = float(request.POST['rainfall'])
        
        # Add these two
        state = request.POST['state']
        city = request.POST['city']
        
        # Assuming you encode state and city into numeric values (You can use Label Encoding)
        # For now, we just convert them to ASCII sum as a placeholder
        state_value = sum([ord(char) for char in state.lower()])
        city_value = sum([ord(char) for char in city.lower()])

        features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, state_value, city_value]
        
        try:
            model = joblib.load('home/models/crop_models.pkl')
            prediction = model.predict([features])[0]
            return render(request, 'home/crop_recommend.html', {'prediction': prediction})
        except Exception as e:
            return HttpResponse(f"Error: {e}")
        
    return render(request, 'home/crop_recommend.html')

def fertilizer_recommendation(request):
    if request.method == "POST":
        try:
            nitrogen = float(request.POST['nitrogen'])
            phosphorus = float(request.POST['phosphorus'])
            potassium = float(request.POST['potassium'])

            # Define normal ranges for soil nutrients (these can be adjusted if needed)
            normal_nitrogen = (80, 120)
            normal_phosphorus = (40, 60)
            normal_potassium = (40, 80)

            # Suggestions based on nutrient levels
            suggestions = []

            if nitrogen < normal_nitrogen[0]:
                suggestions.append("The nitrogen level of your soil is low. Consider using Urea or other nitrogen-rich fertilizers.")
            elif nitrogen > normal_nitrogen[1]:
                suggestions.append("The nitrogen level of your soil is high. Avoid using nitrogen-rich fertilizers.")

            if phosphorus < normal_phosphorus[0]:
                suggestions.append("The phosphorus level of your soil is low. Consider using DAP or organic phosphates.")
            elif phosphorus > normal_phosphorus[1]:
                suggestions.append("The phosphorus level of your soil is high. Avoid using phosphorus-based fertilizers.")

            if potassium < normal_potassium[0]:
                suggestions.append("The potassium level of your soil is low. Try using MOP or natural potash sources like banana peels.")
            elif potassium > normal_potassium[1]:
                suggestions.append("The potassium level of your soil is high. Avoid using potassium-rich fertilizers.")

            # If no issues found, proceed with model prediction
            if not suggestions and fertilizer_model:
                prediction = fertilizer_model.predict([[nitrogen, phosphorus, potassium]])[0]
                fertilizer_description = fertilizer_info.get(prediction, "No information available for this fertilizer.")
            else:
                prediction = "Soil Condition Analysis Complete"
                fertilizer_description = "\n".join(suggestions)

        except Exception as e:
            prediction = f"🚨 Error processing request: {str(e)}"
            fertilizer_description = ""

        return render(request, "fertilizer.html", {
            "recommendation": prediction,
            "description": fertilizer_description
        })

    return render(request, "fertilizer.html")


def disease_detection(request):
    disease_info = {
       
    "tomato_rust": {
        "description": "Tomato rust causes yellow or brown spots on leaves and stems, resulting in poor fruit production.",
        "cure": "Remove infected plants, use fungicides, and plant resistant varieties."
    },
    "tomato_spot": {
        "description": "Tomato spot causes dark spots on fruits and leaves, leading to stunted growth.",
        "cure": "Remove infected plants and use appropriate fungicides."
    },
    "mulberry_rust": {
        "description": "Mulberry rust causes orange rust pustules on leaves, leading to premature leaf drop.",
        "cure": "Apply fungicides and remove infected leaves."
    },
    "Basil_downy mildew": {
        "description": "Basil downy mildew causes yellowing leaves, dark sporulation on the underside, and distorted growth.",
        "cure": "Use resistant varieties, ensure good air circulation, and apply fungicides if needed."
    },
    "banana_rust": {
        "description": "Banana rust causes reddish-brown streaks and spots on leaves, leading to reduced photosynthesis and fruit yield.",
        "cure": "Apply appropriate fungicides and practice proper field sanitation."
    },
    "Apple_black_rot": {
        "description": "Apple black rot results in dark, sunken lesions on fruit, leaves, and branches, often leading to fruit rot and tree decline.",
        "cure": "Prune infected areas, apply fungicides, and maintain orchard cleanliness."
    },
    "grape_rott": {
        "description": "Grape rot causes fruit decay, mold growth, and brown discoloration of the grapes, affecting yield and quality.",
        "cure": "Implement proper pruning, fungicide application, and avoid overhead watering."
    },
    "orange_rust": {
        "description": "Orange rust causes orange powdery spots on leaves, which eventually distort and cause early leaf drop.",
        "cure": "Remove infected plants, apply fungicides, and ensure good air circulation."
    },
    "corn_rust": {
        "description": "Corn rust results in small, reddish-brown pustules on leaves, reducing photosynthesis and overall yield.",
        "cure": "Use resistant varieties, apply fungicides, and maintain crop rotation practices."
    },
    "unknown": {
        "description": "This disease is not recognized. Please consult an agricultural expert.",
        "cure": "No cure information available."
    }
}
    
    context = {}

    if request.method == "POST":
        if 'leaf_image' not in request.FILES:
            context["disease"] = "🚨 No image uploaded!"
            return render(request, "disease.html", context)

        try:
            leaf_image = request.FILES['leaf_image']
            upload_path = os.path.join("home/static/disease_detection", leaf_image.name)

            with open(upload_path, "wb") as f:
                for chunk in leaf_image.chunks():
                    f.write(chunk)

            image = cv2.imread(upload_path)
            if image is None:
                raise ValueError("🚨 Unable to read the uploaded image.")

            features = extract_features(image).reshape(1, -1)

            if disease_model and isinstance(disease_model, tuple) and len(disease_model) == 2:
                clf, label_encoder = disease_model
            else:
                raise ValueError("🚨 Disease model not loaded or incorrect format!")

            prediction = clf.predict(features)[0]
            predicted_disease = label_encoder.inverse_transform([prediction])[0]

            # Get disease details
            disease_details = disease_info.get(predicted_disease, {
                "description": "Description not available.",
                "cure": "Cure information not available."
            })

            # Add the URL of the uploaded image to the context
            context = {
                "disease": predicted_disease,
                "description": disease_details["description"],
                "cure": disease_details["cure"],
                "uploaded_image_url": f"/static/disease_detection/{leaf_image.name}"
            }

        except Exception as e:
            context = {"disease": f"🚨 Error processing request: {str(e)}"}

    return render(request, "disease.html", context)
# Simple Rule-Based Chatbot Functionality

def chatbot_page(request):
    return render(request, 'home/chatbot.html')

# Home Page
def index(request):
    return render(request, "home/index.html")

# Signup Function
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('signup')
        
        user = User.objects.create_user(username=username, password=password, email=email)
        user.save()
        messages.success(request, 'Signup successful. Please login.')
        login(request, user)  # Automatically log the user in after signup
        return redirect('index')  # Redirect to main website page

    return render(request, 'home/signup.html')

# Login Function
# home/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            messages.error(request, "User does not exist.")
            return redirect('login')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to your homepage
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('login')

    return render(request, 'home/login.html')


@login_required
def home_view(request):
    return render(request, 'home/home.html', {'user': request.user})

#logout

def logout_view(request):
    logout(request)
    return redirect('login')


# # Admin Login Function
# def admin_login(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
        
#         user = authenticate(request, username=username, password=password)
        
#         if user is not None and user.is_superuser:
#             login(request, user)
#             return redirect('admin_dashboard')
#         else:
#             messages.error(request, 'Invalid Admin credentials. Please try again.')
#             return redirect('admin_login')
            
#     return render(request, 'home/admin_login.html')

# # Admin Dashboard
# def admin_dashboard(request):
#     if request.user.is_authenticated and request.user.is_superuser:
#         return render(request, 'home/admin_dashboard.html')
#     else:
#         return HttpResponse("Unauthorized Access", status=403)

#ContactUS

def contact_us(request):
    if request.method == 'POST':
        form = ContactUsForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your message has been sent successfully!')
            return redirect('contact_us')  # Redirect after successful submission
    else:
        form = ContactUsForm()

    return render(request, 'home/index.html', {'form': form})
