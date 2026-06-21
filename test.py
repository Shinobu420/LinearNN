import torch
from PIL import Image
from linear_nn import neuralNetwork, transformData

#amd specific
torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transformData(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    
    prediction_index = torch.argmax(output, dim=1).item()
    classes = ["Cat", "Dog"]
    
    print(f"Prediction for {image_path}: {classes[prediction_index]}")


if __name__ == "__main__":
    print(f"Using {device} device")

    model = neuralNetwork().to(device)
    model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
    model.eval()

    predict_image("image.png", model)