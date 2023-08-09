import io
import cv2
import matplotlib
matplotlib.use('Agg')
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

activation = {}


class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super(VGG, self).__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        output = self.classifier(h)
        return output, h


def get_vgg_layers(config, batch_norm):
    """get_vgg_layers iterates over the configuration list and appends each layer to layers"""
    layers: list = []
    in_channels: int = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels: int = c

    return nn.Sequential(*layers)


vgg16_config: list[int | str] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                                 512, 512, 512, 'M', 512, 512, 512, 'M']

vgg16_layers = get_vgg_layers(vgg16_config, batch_norm=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG(vgg16_layers, output_dim=10)
model.load_state_dict(torch.load('prediction_app/model_vgg_pytorch_state_dict.pt', map_location=device))
model: VGG = model.to(device)
model.eval()

'''Composes several transforms together'''
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def classify_image(image_path):
    """image classification"""
    image: Image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    input_batch = image

    model_children = list(model.children())
    model_children[0][0].register_forward_hook(get_activation('cnn_0'))

    with torch.no_grad():
        output = model(input_batch)[0]
        dimensi_iter = activation['cnn_0'][0, :, :, :].size(dim=0)

    heatmap = torch.mean(activation['cnn_0'], dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    activation_img_output = []

    for i in range(3):
        fig = plt.gcf()
        plt.axis('off')
        plt.imshow(activation['cnn_0'][0, i, :, :].cpu().numpy(),
                   cmap=plt.cm.inferno, alpha=.9, interpolation='sinc')
        buf = io.BytesIO()
        fig.savefig(buf, format='JPEG', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        activation_img_output.append(img)

    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    height, width, _ = img.shape

    heatmap = cv2.resize(np.float32(heatmap), (width, height))  # Изменено с... (height, width)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    superimposed_img = heatmap * 0.3 + img

    fig = plt.gcf()
    plt.imshow(superimposed_img.astype(np.uint8) / 255)  # !!!!!

    buf = io.BytesIO()
    fig.savefig(buf, format='JPEG', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    superimposed_img = Image.open(buf)

    _, predicted = torch.max(output, 1)

    output, _ = model(image)
    probabilities = torch.softmax(output, dim=1)

    classes: tuple[str, str, str, str, str, str, str, str, str, str] = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog',
                                                                        'Frog', 'Horse', 'Ship', 'Truck')

    top_probabilities, top_indices = torch.topk(probabilities, k=3, dim=1)
    class_probabilities = [(classes[i], round(float(p) * 100, 2)) for i, p in zip(top_indices[0], top_probabilities[0])]
    predicted_class = classes[top_indices[0][0].item()]

    '''
      Function return:
          1. predicted_class:
              class name ('str')
          2. class_probabilities:
              percent    
          3. activation_img_output:
              LIST of activation pictures (PIL.Image.Image image mode=RGB size=32x32) x3
          4. superimposed_img:
              Gradient-CAM picture (PIL.Image.Image image mode=RGB size=...) x1
      '''

    return predicted_class, class_probabilities, activation_img_output, superimposed_img




