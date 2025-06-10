import torch
import cv2
import numpy as np

def generate_gradcam(model, input_tensor, target_layer, class_idx):
    gradients = []
    activations = []

    def save_gradient_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation_hook(module, input, output):
        activations.append(output)

    handle_grad = target_layer.register_backward_hook(save_gradient_hook)
    handle_act = target_layer.register_forward_hook(save_activation_hook)

    model.zero_grad()
    output = model(input_tensor)
    class_score = output[0, class_idx]
    class_score.backward()

    # Use .detach() to avoid gradient issues
    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (299, 299))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap

def overlay_gradcam(image, heatmap):
    image = np.array(image.resize((299, 299)))
    overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlayed
