import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from config import CNN_MODEL_PATH, DEVICE, CNN_INPUT_SIZE

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        # Front-end: First 10 layers of VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        
        # Back-end: Dilated convolutional layers
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Initialise weights for the back-end
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DensityCNN:
    """
    Wrapper for a CNN-based Density Map Estimator (e.g., CSRNet).
    Predicts a density map where sum(map) == estimated count.
    """
    def __init__(self):
        self.model = self._load_model()
        self.model.to(DEVICE)
        self.model.eval()

    def _load_model(self):
        """Loads the CSRNet architecture and weights."""
        model = CSRNet()
        try:
            if os.path.exists(CNN_MODEL_PATH):
                # Load custom weights if they exist
                checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                return model
            return None
        except Exception as e:
            print(f"Error loading Density CNN: {e}")
            return None

    def predict(self, frame):
        """
        Returns:
            count: float, estimated total count
            density_map: np.ndarray, 2D density map
        """
        if self.model is None:
            return 0.0, None

        # Preprocessing: Normalize and resize
        img = cv2.resize(frame, CNN_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
        
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(img_tensor)
            
            # CSRNet outputs a raw density map (not sigmoid)
            # The sum of all pixels in the map equals the total person count
            density_map = output[0][0].cpu().numpy()
            count = float(np.sum(density_map))
        
        return count, density_map