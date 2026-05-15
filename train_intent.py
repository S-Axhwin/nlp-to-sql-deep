"""
train_intent.py — Fine-tune MiniLM + Linear head on NL→SQL intent dataset.

What happens here (Deep Learning):
  • Forward pass: text → MiniLM embeddings → Linear head → logits
  • Loss:         CrossEntropyLoss (softmax + negative log-likelihood)
  • Backward:     loss.backward() → gradients flow through Linear layers
  • Update:       Adam optimizer steps on Linear weights
  • MiniLM:       frozen (we train only the classification head)

Run:
  python train_intent.py

Output:
  models/intent_classifier.pt  — saved model weights
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.intent_dataset import DATASET, LABEL2ID, ID2LABEL, INTENT_LABELS
from backend.nlp.intent_model import IntentClassifier

# ── Config ──────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 8
LEARNING_RATE = 2e-4
HIDDEN_DIM    = 128
DROPOUT       = 0.1
SAVE_PATH     = "models/intent_classifier.pt"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────


class IntentDataset(Dataset):
    def __init__(self, data):
        self.texts  = [text for text, _ in data]
        self.labels = [LABEL2ID[label] for _, label in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


def train():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {len(DATASET)} examples, {len(INTENT_LABELS)} classes")
    print(f"Classes: {INTENT_LABELS}\n")

    # ── Model ──
    model = IntentClassifier(
        num_classes=len(INTENT_LABELS),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    # Freeze MiniLM encoder — only train the classification head
    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {trainable:,}")
    print(f"Total parameters     : {total:,}")
    print(f"Frozen (MiniLM)      : {total - trainable:,}\n")

    # ── Data ──
    dataset    = IntentDataset(DATASET)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # ── Loss + Optimizer ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # ── Training Loop ──
    print("=" * 55)
    print(f"{'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        total_samples = 0

        for texts, labels in dataloader:
            labels = labels.to(DEVICE)

            # Forward pass
            embeddings = model.encode_text(texts, DEVICE)  # (B, 384)
            logits     = model(embeddings)                  # (B, 7)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(texts)
            preds          = torch.argmax(logits, dim=-1)
            correct       += (preds == labels).sum().item()
            total_samples += len(texts)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples * 100

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg_loss:>10.4f}  {accuracy:>9.1f}%")

    print("=" * 55)

    # ── Final evaluation (same data — small dataset sanity check) ──
    model.eval()
    print("\nFinal predictions on sample queries:")
    samples = [
        "how many students are there",
        "show top 5 students by cgpa",
        "students with attendance below 75",
        "average cgpa of all students",
        "list all CSE students",
        "students with cgpa above 8",
        "worst 3 students by marks",
    ]
    for text in samples:
        label_id, conf = model.predict(text, DEVICE)
        print(f"  [{ID2LABEL[label_id]:12s} {conf*100:5.1f}%]  {text}")

    # ── Save ──
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state_dict" : model.state_dict(),
        "label2id"         : LABEL2ID,
        "id2label"         : ID2LABEL,
        "hidden_dim"       : HIDDEN_DIM,
        "num_classes"      : len(INTENT_LABELS),
    }, SAVE_PATH)
    print(f"\nModel saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
