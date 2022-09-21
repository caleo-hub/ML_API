import sys
sys.path.append("C:\\Users\csant321\Desktop\Projetos\ML_API")
from model.neuralNetwork import NeuralNetwork
import torch
import torchvision.transforms as transforms
from PIL import Image
import io


channel_size = 1
img_size = (28,28)


# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load("data/model.pth"))
model.eval()
# image -> tensor

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

#predict
def get_prediction(image_tensor):
     
    images = image_tensor.reshape(-1, *img_size)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted