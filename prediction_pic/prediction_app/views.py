import base64
import copy
import io
import json
import os
import random

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from .vgg_classification import classify_image


def image_to_base64(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='JPEG')
    img_bytes = img_byte_array.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


def home(request):
    """send a request POST and get a response"""
    if request.method == 'POST':
        image = request.FILES.get('image')
        copy_image = copy.deepcopy(image)

        if image:
            predicted_class, class_probabilities, activation_img_output, gradient = classify_image(image)
            encoded_image: str = base64.b64encode(copy_image.read()).decode('utf-8')
            activation_img_base64 = [image_to_base64(img) for img in activation_img_output]
            gradient_base64 = image_to_base64(gradient)

            json_path: str = os.path.join(settings.BASE_DIR, 'prediction_app', 'static', 'json', 'facts.json')
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
            facts_list = data[predicted_class]["facts"]
            fact = random.choices(facts_list, k=1)[0]

            context = {
                'predict_res': predicted_class,
                'probability': class_probabilities,
                'image': encoded_image,
                'fact': fact,
                'activation_img': activation_img_base64,
                'gradient': gradient_base64
            }

            return render(request, 'prediction_app/home.html', context=context)
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
    else:
        return render(request, 'prediction_app/home.html')
