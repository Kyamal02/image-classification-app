import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO


@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model


@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return [line.strip().decode('utf-8') for line in BytesIO(response.content).readlines()]


def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def main():
    st.title("Классификация изображений")
    st.write("Загрузите изображение для классификации")

    model = load_model()
    labels = load_labels()
    preprocess = get_preprocess_transform()

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Загруженное изображение", use_container_width=True)

            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)

            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_probs, top5_classes = torch.topk(probabilities, 5)

            st.subheader("Топ-5 предсказаний:")
            for i in range(top5_probs.size(0)):
                st.write(
                    f"{i + 1}. {labels[top5_classes[i]]} "
                    f"({top5_probs[i].item() * 100:.2f}%)"
                )

        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")


if __name__ == "__main__":
    main()
