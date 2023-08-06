import pytest
from prediction_app.vgg_classification import *
from prediction_app.views import *
import requests


def test_get_vgg_layers():
    result = get_vgg_layers(vgg16_config, True)
    assert type(result) is torch.nn.modules.container.Sequential


def test_classify_image_dog():
    result = classify_image('prediction_app/images/dog.jfif')
    assert type(result) is tuple
    assert type(result[0]) is str
    assert result[0] is 'Dog'


def test_classify_image_cat():
    result = classify_image('prediction_app/images/cat.jfif')
    assert type(result) is tuple
    assert type(result[0]) is str
    assert result[0] is 'Cat'


def test_classify_image_horse():
    result = classify_image('prediction_app/images/horse.jfif')
    print(result[0])
    assert type(result) is tuple
    assert type(result[0]) is str
    assert result[0] is 'Horse'


def test_classify_image_deer():
    result = classify_image('prediction_app/images/deer.jfif')
    assert type(result) is tuple
    assert type(result[0]) is str
    assert result[0] is 'Deer'


def test_classify_image_frog():
    result = classify_image('prediction_app/images/frof.jfif')
    assert type(result) is tuple
    assert type(result[0]) is str
    assert result[0] is 'Frog'


def test_vgg_class():
    mVGG = VGG(vgg16_layers, output_dim=10)
    assert type(mVGG) is VGG
