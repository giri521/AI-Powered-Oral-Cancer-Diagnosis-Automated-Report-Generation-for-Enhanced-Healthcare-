# train_convnext.py
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import sys
import os

# ========== Input ==========
img_type = sys.argv[1] if len(sys.argv) > 1 else "clinical"
data_dir = f"data/{img_type}"
model_path = f"model/convnext_{img_type}.pt"

# ========== Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training ConvNeXt-Tiny on '{img_type}' images")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = models.convnext_tiny(weights='IMAGENET1K_V1')
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ========== Train ==========
for epoch in range(5):
    total, correct, loss_sum = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        loss_sum += loss.item()

    print(f"Epoch [{epoch+1}/5] Loss: {loss_sum:.4f} Accuracy: {100*correct/total:.2f}%")

# ========== Save ==========
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✅ Saved: {model_path}")
