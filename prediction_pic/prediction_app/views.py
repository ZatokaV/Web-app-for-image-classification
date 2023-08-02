import base64
import copy

from django.http import JsonResponse
from django.shortcuts import render
from .vgg_classification import classify_image


def home(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        copy_image = copy.deepcopy(image)

        if image:
            result = classify_image(image)
            encoded_image = base64.b64encode(copy_image.read()).decode('utf-8')

            context = {
                'predict_res': result,
                'image': encoded_image,
            }

            return render(request, 'prediction_app/home.html', context=context)
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
    else:
        return render(request, 'prediction_app/home.html')