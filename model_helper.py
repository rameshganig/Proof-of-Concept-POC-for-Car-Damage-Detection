from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

trained_model = None
class_names = ['Front_Breakage', 'Front_Crushed', 'Front_Normal', 'Rear_Breakage', 'Rear_Crushed', 'Rear_Normal']

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # freezing all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreezing layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # FIX: Change .features to .classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.49),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path):
    # 1. Standardize the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model
    if trained_model is None:
        # Pass len(class_names) which is 6
        # Make sure variable name is EXACTLY 'trained_model'
        trained_model = CarClassifierResNet(num_classes=len(class_names))

        # Load weights - added map_location for compatibility
        state_dict = torch.load(r"model\saved_model.pth", map_location=torch.device('cpu'))
        trained_model.load_state_dict(state_dict)
        trained_model.eval()

    # 2. Run inference
    with torch.no_grad():
        # This will now work because trained_model is globally defined above
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
