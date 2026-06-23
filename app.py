import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import tifffile as tiff
import numpy as np

st.set_page_config(
    page_title="GeoSnap",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoSnap")
st.subheader("Satellite Image Classification")

# ===========================
# MODEL CREATION
# ===========================

@st.cache_resource
def load_rgb_model():

    checkpoint = torch.load(
        "efficientnet_b2_rgb_final.pth",
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    model = models.efficientnet_b2(weights=None)

    n_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        n_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, classes


@st.cache_resource
def load_ms_model():

    checkpoint = torch.load(
        "efficientnet_b2_ms_final.pth",
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    band_mean = checkpoint["band_mean"]
    band_std = checkpoint["band_std"]

    model = models.efficientnet_b2(weights=None)

    old_conv = model.features[0][0]

    new_conv = nn.Conv2d(
        12,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    model.features[0][0] = new_conv

    n_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        n_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, classes, band_mean, band_std


# ===========================
# IMAGE TRANSFORMS
# ===========================

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# UI
# ===========================

model_choice = st.selectbox(
    "Select Input Type",
    [
        "RGB Image (.jpg/.png)",
        "Multispectral GeoTIFF (.tif)"
    ]
)

if model_choice == "RGB Image (.jpg/.png)":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded_file = st.file_uploader(
        "Upload GeoTIFF",
        type=["tif", "tiff"]
    )

# ===========================
# PREDICTION
# ===========================

if uploaded_file is not None:

    if model_choice == "RGB Image (.jpg/.png)":

        model, classes = load_rgb_model()

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        x = rgb_transform(image)
        x = x.unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)

    else:

        (
            model,
            classes,
            band_mean,
            band_std
        ) = load_ms_model()

        img = tiff.imread(uploaded_file)

        img = img.astype(np.float32)

        img = (
            img - np.array(band_mean)[:, None, None]
        ) / (
            np.array(band_std)[:, None, None] + 1e-8
        )

        x = torch.tensor(img).float().unsqueeze(0)

        st.success(
            f"Loaded TIFF with shape: {img.shape}"
        )

        with torch.no_grad():
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item()

    st.success(
        f"Prediction: {classes[pred]}"
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.subheader("Top Predictions")

    top_probs, top_idx = torch.topk(
        probs,
        min(3, len(classes))
    )

    for p, i in zip(
        top_probs[0],
        top_idx[0]
    ):

        st.write(
            f"{classes[i]} : {p.item()*100:.2f}%"
        )

    chart_data = {
        classes[i]:
        probs[0][i].item()
        for i in range(len(classes))
    }

    st.bar_chart(chart_data)