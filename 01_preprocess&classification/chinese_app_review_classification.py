import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn as nn


class ChineseReviewClassifier(nn.Module):
    def __init__(self, model_name="../chinese-roberta-wwm-ext-large"):
        super(ChineseReviewClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def predict(self, logits):
        predictions = (torch.sigmoid(logits) > 0.5).float()
        return predictions


class ChineseReviewDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)  # 单标签，需要unsqueeze

        return item


def train_model(model, train_dataloader, val_dataloader, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        val_metrics = evaluate_model(model, val_dataloader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Metrics: Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = model.predict(outputs.logits).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def predict_reviews(model, tokenizer, texts, device, max_length=128):
    model.eval()
    dataset = ChineseReviewDataset(texts, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=16)
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = model.predict(outputs.logits).cpu().numpy()
            all_predictions.extend(predictions)

    return np.array(all_predictions).flatten()
