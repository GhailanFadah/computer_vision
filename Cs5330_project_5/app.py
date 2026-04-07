"""
Ghailan Fadah
CS5530
4/6/26
Live digit recognition app using the trained MNIST CNN and Gradio.
Accepts a webcam image, preprocesses it to match MNIST, and returns the predicted digit.
"""

import numpy as np
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image, ImageOps

from my_cnn import DigitRecognitionNet


def load_model(path='model.pth'):
    """Load the trained MNIST model."""
    model = DigitRecognitionNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Convert a webcam frame (RGB numpy array) to a normalized MNIST-style tensor.
    - Grayscale
    - Resize to 28x28
    - Invert (MNIST digits are white on black)
    - Normalize with MNIST mean/std
    """
    pil = Image.fromarray(image).convert('L')
    pil = pil.resize((28, 28), Image.LANCZOS)
    pil = ImageOps.invert(pil)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(pil).unsqueeze(0)  # (1, 1, 28, 28)


MODEL = load_model('model.pth')


def predict(image: np.ndarray):
    """Run the model on the webcam frame and return the predicted digit and probabilities."""
    if image is None:
        return 'No image', {}

    tensor = preprocess(image)

    with torch.no_grad():
        probs = MODEL(tensor).exp().squeeze()

    predicted = probs.argmax().item()
    confidence = {str(i): round(probs[i].item(), 4) for i in range(10)}

    return f'Predicted digit: {predicted}', confidence


with gr.Blocks(title='MNIST Digit Recognition') as demo:
    gr.Markdown('## Live Digit Recognition\nHold a handwritten digit up to your webcam.')

    with gr.Row():
        webcam = gr.Image(sources='webcam', streaming=True, label='Webcam')
        with gr.Column():
            result = gr.Textbox(label='Prediction', interactive=False)
            probs  = gr.Label(num_top_classes=10, label='Class Probabilities')

    webcam.stream(fn=predict, inputs=webcam, outputs=[result, probs])


if __name__ == '__main__':
    demo.launch()
