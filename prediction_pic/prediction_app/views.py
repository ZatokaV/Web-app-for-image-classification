from django.http import JsonResponse
from django.shortcuts import render
from .vgg_classification import classify_image


def home(request):
    if request.method == 'POST':
        image = request.FILES.get('image')

        if image:
            result = classify_image(image)

            return JsonResponse({'predicted_class': result})
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
    else:
        return render(request, 'prediction_app/home.html')