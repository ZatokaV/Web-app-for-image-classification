import base64
import copy
import json
import os
import random

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from .vgg_classification import classify_image


def home(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        copy_image = copy.deepcopy(image)

        if image:
            predicted_class, class_probabilities = classify_image(image)
            encoded_image = base64.b64encode(copy_image.read()).decode('utf-8')

            json_path = os.path.join(settings.BASE_DIR, 'prediction_app', 'static', 'json', 'facts.json')
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
            facts_list = data[predicted_class]["facts"]
            fact = random.choices(facts_list, k=1)[0]

            context = {
                'predict_res': predicted_class,
                'probability': class_probabilities,
                'image': encoded_image,
                'fact': fact
            }

            return render(request, 'prediction_app/home.html', context=context)
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
    else:
        return render(request, 'prediction_app/home.html')
