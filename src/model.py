"""model.py — NurseXAIBERT architecture"""
import torch
import torch.nn as nn
from transformers import BertModel


class NurseXAIBERT(nn.Module):
    """BERTimbau fine-tuned for multi-label NANDA-I classification.
    Architecture: BertModel → CLS token → Dropout → Linear(768, n_labels)
    """
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert       = BertModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))
