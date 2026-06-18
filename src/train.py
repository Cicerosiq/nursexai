"""train.py — Dataset, training loop and K-Fold stratified training"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from model import NurseXAIBERT


class NotasEnfermagemDataset(Dataset):
    """PyTorch Dataset — tokenises internally in __getitem__."""
    def __init__(self, textos, labels, tokenizer, max_length: int = 256):
        self.textos     = list(textos)
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.textos[idx], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels'        : torch.FloatTensor(self.labels[idx])
        }


def calibrar_thresholds(probs_val, labels_val, label_names,
                         candidatos=None):
    """Grid search for optimal per-label threshold on validation set."""
    if candidatos is None:
        candidatos = np.arange(0.10, 0.91, 0.05)
    thresholds = {}
    for idx, label in enumerate(label_names):
        melhor_t, melhor_f1 = 0.5, 0.0
        for t in candidatos:
            preds = (probs_val[:, idx] >= t).astype(int)
            f1    = f1_score(labels_val[:, idx], preds, zero_division=0)
            if f1 > melhor_f1:
                melhor_f1, melhor_t = f1, t
        thresholds[label] = round(float(melhor_t), 2)
    return thresholds
