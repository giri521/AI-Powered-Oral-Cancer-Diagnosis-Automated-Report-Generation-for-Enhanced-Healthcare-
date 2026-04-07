# train_squeezenet.py
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import sys, os

img_type = sys.argv[1] if len(sys.argv) > 1 else "clinical"
data_dir = f"data/{img_type}"
model_path = f"model/squeezenet_{img_type}.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training SqueezeNet1_0 on '{img_type}' dataset")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = models.squeezenet1_0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=1)
model.num_classes = 2
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    total = correct = loss_sum = 0
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

    print(f"[{epoch+1}/5] Loss:{loss_sum:.4f} Acc:{correct/total*100:.2f}%")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Saved: {model_path}")
