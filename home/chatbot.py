import random
from django.http import JsonResponse

class SimpleChatbot:
    def __init__(self):
        self.responses = {
            'hello': ['Hi there!', 'Hello!', 'Hey! How can I assist you?'],
            'hi': ['Hello!', 'Hi! How can I help you today?'],
            'how are you': ['I am just a bunch of code, but I am here to help you!', 'Doing great! How can I assist you?'],
            'crop recommendation': ['Sure, I can help you with crop recommendation. Please provide the details like soil type, temperature, etc.'],
            'fertilizer recommendation': ['I can assist you with fertilizer recommendations. Please provide the crop details.'],
            'disease detection': ['For disease detection, please upload an image of the plant or describe the symptoms.'],
            'bye': ['Goodbye! Have a great day!', 'See you later! Take care.']
        }

    def get_response(self, user_input):
        user_input = user_input.lower()
        for key in self.responses.keys():
            if key in user_input:
                return random.choice(self.responses[key])
        return "I'm sorry, I don't understand. Can you please elaborate?"


# Django View for Chatbot
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json

chatbot = SimpleChatbot()

@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data.get('message', '')
        response = chatbot.get_response(user_input)
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)
