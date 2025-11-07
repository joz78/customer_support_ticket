# safe_training.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple model
class SimpleDistilBERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# Dataset
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_and_save_safely():
    print(" Training with safe saving...")
    
    # Load data
    df = pd.read_csv('customer_support_tickets_cleaned.csv')
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'] != 'Unknown']
    df = df[df['label'] != 'Unknown']
    
    print(f"Training samples: {len(df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Split data
    texts = df['text'].values
    labels = df['encoded_label'].values
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    num_classes = len(label_encoder.classes_)
    model = SimpleDistilBERT(num_classes=num_classes)
    model = model.to(device)
    
    # Simple training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TicketDataset(train_texts, train_labels, tokenizer)
    val_dataset = TicketDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Train for a few epochs
    best_accuracy = 0
    for epoch in range(3):
        print(f'Epoch {epoch + 1}/3')
        
        # Training
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            # SAFE SAVING: Separate model weights and metadata
            # 1. Save model weights only (safe)
            torch.save(model.state_dict(), 'models/model_weights.pth')
            
            # 2. Save metadata separately (safe)
            metadata = {
                'classes': label_encoder.classes_.tolist(),
                'class_to_idx': {cls: idx for idx, cls in enumerate(label_encoder.classes_)},
                'accuracy': accuracy,
                'tokenizer_name': 'distilbert-base-uncased',
                'max_length': 128
            }
            
            with open('models/model_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 Saved model with accuracy: {accuracy:.4f}")
    
    print("✅ Training completed safely!")

if __name__ == "__main__":
    train_and_save_safely()