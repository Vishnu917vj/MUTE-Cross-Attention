import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, ViTModel, AutoTokenizer, AutoModel
from tqdm import tqdm

# Define the CrossAttention module as provided
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, q, kv):
        # q = [batch, q_len, dim]
        # kv = [batch, kv_len, dim]
        out, _ = self.attn(q, kv, kv)
        return out

# Define the CrossAttentionFusion module as provided
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        self.img_to_txt_attn = CrossAttention(dim)
        self.txt_to_img_attn = CrossAttention(dim)

        self.fc = nn.Sequential(
            nn.Linear(dim*2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )

    def forward(self, img_tokens, txt_tokens):
        # Cross attentions
        i2t = self.img_to_txt_attn(img_tokens, txt_tokens)  # image attends to text
        t2i = self.txt_to_img_attn(txt_tokens, img_tokens)  # text attends to image

        # Pooling
        img_context = i2t.mean(dim=1)
        txt_context = t2i.mean(dim=1)

        fused = torch.cat([img_context, txt_context], dim=1)

        return self.fc(fused)

# Custom Dataset for loading data and extracting features
class MemeDataset(Dataset):
    def __init__(self, csv_file, vision_model, text_model, image_processor, tokenizer, device):
        self.data = pd.read_excel(csv_file)
        self.data.columns = self.data.columns.str.strip()

        self.data['image_path'] = self.data['image_path'].astype(str).str.strip()
        self.data['transcript'] = self.data['transcript'].astype(str)
        self.data['target'] = self.data['target'].astype(str)

        self.vision_model = vision_model
        self.text_model = text_model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = str(item['image_path']).strip()
        image = Image.open(image_path).convert('RGB')
        image_inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_outputs = self.vision_model(**image_inputs)
            img_tokens = img_outputs.last_hidden_state  # [1, 197, 768]

        # Process transcript and target separately
        transcript = item['transcript']
        target = item['target']
        
        trans_inputs = self.tokenizer(transcript, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(self.device)
        targ_inputs = self.tokenizer(target, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            trans_outputs = self.text_model(**trans_inputs)
            targ_outputs = self.text_model(**targ_inputs)
            txt_tokens = torch.cat((trans_outputs.last_hidden_state, targ_outputs.last_hidden_state), dim=1)  # [1, 512, 768]

        # Assume stance is already an integer (0,1,2); if string, add mapping here
        label = int(item['stance'])

        return img_tokens.squeeze(0), txt_tokens.squeeze(0), label

# Main training script
def train_model(csv_file='data.csv', batch_size=8, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "meme_stance_model.pth"
    print(f"Using device: {device}")

    # Load pre-trained models for feature extraction
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    text_model  = AutoModel.from_pretrained("google/muril-base-cased").to(device)
    text_model.eval()
    for param in text_model.parameters():
        param.requires_grad = False

    # Dataset and DataLoader
    dataset = MemeDataset(csv_file, vision_model, text_model, image_processor, tokenizer, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, loss
    model = CrossAttentionFusion(dim=768).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img_tokens, txt_tokens, labels in tqdm(dataloader):
            img_tokens = img_tokens.to(device)
            txt_tokens = txt_tokens.to(device)
            labels = torch.tensor(labels).to(device)  # Ensure labels are tensor

            optimizer.zero_grad()
            outputs = model(img_tokens, txt_tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'meme_stance_model.pth')
    print("Training completed. Model saved as 'meme_stance_model.pth'.")

# Run the training (adjust csv_file if needed) 
if __name__ == "__main__": 
    train_model(csv_file='c:\\Users\\RGUKT\\Downloads\\Temp\\memes_january\\train_feb2.xlsx')  # Replace 'data.csv' with your actual CSV file path
    