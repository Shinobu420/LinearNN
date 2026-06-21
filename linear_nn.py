import torch
from torch import nn
from torchvision.transforms import v2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # flatten image to pixel row (1D "Tensor")
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 256*256*3 compressed down to 512 features
            nn.Linear(256 * 256 * 3, 512),
            nn.ReLU(),
            # hidden layer with 512 features
            nn.Linear(512, 512),
            nn.ReLU(),
            # output layer with 2 features (cat / dog)
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

transform_data = v2.Compose([
    # resize every image to 256x256
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
