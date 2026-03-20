import os
import torch
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

data = pd.read_csv("static/data/prompt.csv")

label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["sketch_image"])

num_classes = len(label_encoder.classes_)

# ------------------------------------------------
# Image Transform
# ------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ------------------------------------------------
# Dataset Class
# ------------------------------------------------

class CrimeDataset(Dataset):

    def __init__(self, dataframe):

        self.data = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        text = row["prompt"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        img_path = os.path.join("dataset/sketches", row["sketch_image"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)

        return input_ids, attention_mask, image, label


# ------------------------------------------------
# Text Model
# ------------------------------------------------

text_model = AutoModel.from_pretrained("distilbert-base-uncased")

# ------------------------------------------------
# Image Model
# ------------------------------------------------

image_model = models.resnet18(weights="DEFAULT")
image_model.fc = nn.Linear(512,128)

# ------------------------------------------------
# Fusion Model
# ------------------------------------------------

class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_model = text_model
        self.image_model = image_model

        self.fc = nn.Linear(768 + 128, num_classes)

    def forward(self, input_ids, attention_mask, image):

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = text_output.last_hidden_state[:,0,:]

        img_features = self.image_model(image)

        combined = torch.cat((text_features, img_features), dim=1)

        output = self.fc(combined)

        return output


# ------------------------------------------------
# Training Function
# ------------------------------------------------

def train():

    dataset = CrimeDataset(data)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = FusionModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):

        model.train()

        total_loss = 0

        for input_ids, attention_mask, image, label in loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(input_ids, attention_mask, image)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/crime_model.pth")

    print("Training completed successfully")


# ------------------------------------------------
# Run Training Only When File Executed Directly
# ------------------------------------------------

if __name__ == "__main__":
    train()
