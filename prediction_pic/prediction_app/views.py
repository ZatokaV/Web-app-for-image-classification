from django.shortcuts import render
from .forms import ImageUploadForm


def classify_image(image):
    pass


def home(request):
    result = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']

            result = classify_image(image)
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'result': result
    }

    return render(request, 'prediction_app/home.html', context)
