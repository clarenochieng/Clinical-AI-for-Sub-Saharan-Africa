import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

import pandas as pd
import streamlit as st

from main import predict_disease
from utils import find_unique_values

st.title("Health App")

sideBarHomePageChoice = st.sidebar.selectbox("Select a feature: ", ("Symptom Lookup", "Malaria Image Analysis"))

if sideBarHomePageChoice == "Symptom Lookup":
    predictedDiagnosis = None
    symptomsList = find_unique_values(pd.read_csv("dataset.csv"))
    symptomsList.pop(0)
    symptomsChoice = st.multiselect("Enter Patient Symptoms: ", symptomsList)
    predictedDiagnosis = predict_disease(symptomsChoice)
    if len(symptomsChoice) < 3:
        st.write("Not enough symptoms selected")
    else:
        st.write(predictedDiagnosis)

if sideBarHomePageChoice == "Malaria Image Analysis":
    @st.cache_data
    def load_model():
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 2  # 1 class (malaria parasite) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load('fasterrcnn_model_state.pth', map_location=torch.device('cpu')))
        model.eval()
        return model

    def preprocess_image(image):
        image_tensor = F.to_tensor(image).unsqueeze(0)
        return image_tensor

    def annotate_image(image, model):
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            predictions = model(image_tensor)

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        for score, box in zip(predictions[0]['scores'], predictions[0]['boxes']):
            if score > 0.2:  # Adjust the threshold as needed
                draw.rectangle(box.tolist(), outline="red", width=2)
        return annotated_image

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        model = load_model()
        image = Image.open(uploaded_image).convert("RGB")
        annotated_image = annotate_image(image, model)
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
