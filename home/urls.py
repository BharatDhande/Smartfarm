from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_view, name='login'),
    path('', views.home_view, name='home'),
    path('logout/', views.logout_view, name='logout'),
    path('train_crop_model/', views.train_crop_model, name='train_crop_model'),
    path('crop_recommend/', views.crop_recommend, name='crop_recommend'),
    path('fertilizer/', views.fertilizer_recommendation, name='fertilizer_recommendation'),
    path('disease/', views.disease_detection, name='disease_detection'), 
    # path('chat/', views.chatbot_response, name='chatbot_response'),
    path('contact/', views.contact_us, name='contact_us'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('chat/', views.chatbot_page, name='chatbot_page'),
    
    # path('predict_crop/', views.predict_crop, name='predict_crop'),
]
