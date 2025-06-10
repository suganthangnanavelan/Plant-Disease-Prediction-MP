import streamlit as st
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import torch
import matplotlib.pyplot as plt
from utils import load_model, preprocess_image, predict
from gradcam import generate_gradcam, overlay_gradcam

# Set page config
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Custom styling
st.markdown("""
    <style>
    body { background-color: #4CAF50; color: #4CAF50; }
    .main { background-color: #4CAF50; }
    .header { background-color: #4CAF50; padding: 1.5rem; text-align: center;
              font-size: 2rem; font-weight: bold; color: #ffffff; letter-spacing: 1px; }
    .left, .right { padding: 2rem; color: #4CAF50; }
    .left img { width: 150px; margin-bottom: 1rem; }
    .feature { margin: 1rem 0; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">PLANT DISEASE PREDICTION</div>', unsafe_allow_html=True)

# Mapping of plant type to model path and class names
plant_models = {
    "Apple": {
        "model_path": "models/inception_v3_model_apple.pth",
        "class_names": ['Black_rot', 'Healthy', 'Rust', 'Scab']
    },
    "Pepper": {
        "model_path": "models/inception_v3_model_pepper.pth",
        "class_names": ['Bacterial_spot', 'Healthy']
    },
    "Corn": {
        "model_path": "models/inception_v3_model_corn.pth",
        "class_names": ['Cercospora_leaf_spot', 'Common_rust', 'Healthy', 'Northern_leaf_blight']
    },
    "Potato": {
        "model_path": "models/inception_v3_model_potato.pth",
        "class_names": ['Early_blight', 'Healthy', 'Late_blight']
    }
}

# Upload Section
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="left">', unsafe_allow_html=True)

    # Display animated GIF centered with rounded corners using inline CSS
    gif_path = "groot.gif"
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
    gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
    gif_html = f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/gif;base64,{gif_b64}" width="240" 
                 style="border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);" 
                 alt="Animated Groot GIF">
        </div>
    """
    st.markdown(gif_html, unsafe_allow_html=True)

    st.markdown("""
        <h4 style="color:#4CAF50;">Is your plant affected by an unknown disease?</h4>
        <h5 style="color:#4CAF50;">Don't worry!</h5>
        <div class="feature">1️⃣ Upload the infected part of your plant</div>
        <div class="feature">2️⃣ Select the plant type</div>
        <div class="feature">3️⃣ Take necessary actions based on diagnosis</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#4CAF50;">Upload Plant Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose File", type=["jpg", "jpeg", "png"])
    plant_type = st.selectbox("Select Plant Type", list(plant_models.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

# Load the appropriate model and class names
model_info = plant_models[plant_type]
model = load_model(model_info["model_path"])
class_names = model_info["class_names"]

def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_feature_maps(model, image_tensor, layers):
    fmap_dict = {}
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            fmap_dict[name] = output.detach().cpu().squeeze(0)
        return hook_fn

    for name, layer in layers.items():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    model(image_tensor)
    for h in hooks:
        h.remove()

    return fmap_dict

def display_image_with_download_button(image_b64, filename, icon_b64):
    st.markdown(f"""
        <div style="position: relative; display: inline-block;">
            <img src="data:image/png;base64,{image_b64}" style="width: 300px; border-radius: 10px;" />
            <a href="data:image/png;base64,{image_b64}" download="{filename}"
               style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.7); padding: 5px; border-radius: 5px;">
                <img src="data:image/png;base64,{icon_b64}" style="width: 16px;" />
            </a>
        </div>
    """, unsafe_allow_html=True)

# 🔍 Prediction + Visualization
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((300, 300))
    image_tensor = preprocess_image(image)
    prediction, probs = predict(image_tensor, model, class_names)

    pred_color = "#32CD32" if prediction == "Healthy" else "#FF5733"
    st.markdown(f"<h3 style='text-align: center; color: {pred_color}; margin-bottom: 1rem;'>Prediction: {prediction}</h3>", unsafe_allow_html=True)

    # Grad-CAM
    target_layer = model.Mixed_7c
    class_idx = np.argmax(probs)
    heatmap = generate_gradcam(model, image_tensor.detach(), target_layer, class_idx)
    gradcam_overlay = overlay_gradcam(image, heatmap)
    gradcam_resized = Image.fromarray(gradcam_overlay).resize((300, 300))

    # Convert images to base64
    image_b64 = image_to_base64(image_resized)
    gradcam_b64 = image_to_base64(gradcam_resized)

    # Load download icon
    with open("download_icon.png", "rb") as f:
        icon_b64 = base64.b64encode(f.read()).decode()

    # Show images with download
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Uploaded Leaf**")
        display_image_with_download_button(image_b64, "uploaded_leaf.png", icon_b64)

    with col_img2:
        st.markdown("**Grad-CAM Overlay**")
        display_image_with_download_button(gradcam_b64, "gradcam_overlay.png", icon_b64)

    # 🔬 Feature Maps
    st.markdown("### Feature Maps")
    layers_to_extract = {
        "Conv2d_1a_3x3": model.Conv2d_1a_3x3,
        "Conv2d_2a_3x3": model.Conv2d_2a_3x3,
        "Conv2d_2b_3x3": model.Conv2d_2b_3x3,
        "Conv2d_3b_1x1": model.Conv2d_3b_1x1,
        "Conv2d_4a_3x3": model.Conv2d_4a_3x3,
        "Mixed_5d": model.Mixed_5d,
        "Mixed_6e": model.Mixed_6e,
        "Mixed_7c": model.Mixed_7c
    }
    feature_maps = extract_feature_maps(model, image_tensor, layers_to_extract)

    for layer_name, fmap in feature_maps.items():
        st.markdown(f"**{layer_name}**")
        num_maps = min(8, fmap.shape[0])
        fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
        for i in range(num_maps):
            axes[i].imshow(fmap[i], cmap='viridis')
            axes[i].axis("off")
        st.pyplot(fig)
