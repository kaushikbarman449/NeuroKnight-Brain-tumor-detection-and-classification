from flask import Flask, render_template, request
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from torchinfo import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from going_modular.going_modular import engine, utils
from going_modular.going_modular.predictions import pred_and_plot_image
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():

    # get the Patient's name and save the image
    patientName = request.form['patientName']
    imagefile = request.files['imagefile']
    patientFolder_path = "./images/" + patientName

    if not os.path.exists(patientFolder_path):
        os.makedirs(patientFolder_path)
    image_path = os.path.join(patientFolder_path, imagefile.filename)
    imagefile.save(image_path)

    # image preprocessing
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    final_image = image.resize((224, 224))
    preprocessedImage_path = os.path.join(
        patientFolder_path, 'preprocessed.jpg')
    final_image.save(preprocessedImage_path)

    # ML model
    device = 'cpu'
    set_seeds()

    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    pretrained_vit = torchvision.models.vit_b_16(
        weights=pretrained_vit_weights).to(device)

    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    class_names = ['glioma_tumor', 'meningioma_tumor',
                   'no_tumor', 'pituitary_tumor']
    pretrained_vit.heads = nn.Linear(
        in_features=768, out_features=len(class_names)).to(device)

    saved_model_path = "models/05_going_modular_script_mode_tinyvgg_model.pth"
    saved_model_state_dict = torch.load(
        saved_model_path, map_location=torch.device('cpu'))

    pretrained_vit.load_state_dict(saved_model_state_dict)

    pretrained_vit.eval()

    modelResults_path = './static/' + patientName
    if not os.path.exists(modelResults_path):
        os.makedirs(modelResults_path)
    save_path = os.path.join(modelResults_path, 'output.jpg')

    pred_and_plot_image(model=pretrained_vit,
                        image_path=preprocessedImage_path,
                        class_names=class_names,
                        save_path=save_path)

    return render_template('index.html', save_path=save_path, patientName=patientName)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
