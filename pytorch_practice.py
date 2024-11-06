import torch.nn as nn
import sys

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.layer1 = nn.Sequential(  # Top-level layer containing nested layers
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Linear(32 * 28 * 28, 128)  # Top-level layer
        self.layer3 = nn.Linear(128, 10)  # Top-level layer

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = ComplexModel()

print(model)

sys.exit()
for name, layer in model.named_modules():
    print(f"{name}: {layer}")