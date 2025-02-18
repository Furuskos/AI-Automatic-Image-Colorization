import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained DeOldify model
model = torch.hub.load("jantic/DeOldify", "deoldify", pretrained=True)

def colorize_image(image_path):
    """Colorizes a black & white image."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    # Process image with DeOldify
    output = model(img_tensor)
    output_img = transforms.ToPILImage()(output.squeeze(0))
    output_img.show()

if __name__ == "__main__":
    colorize_image("black_and_white.jpg")
