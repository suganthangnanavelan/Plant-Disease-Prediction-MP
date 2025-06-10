import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes=4):
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if model.AuxLogits:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def preprocess_image(image):
    input_size = 299
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(image_tensor, model, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        return class_names[pred_idx], probs
