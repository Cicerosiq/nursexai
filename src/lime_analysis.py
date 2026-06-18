"""lime_analysis.py — LIME XAI with CLIX-M systematic evaluation"""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from lime.lime_text import LimeTextExplainer

STOPWORDS_PT = {
    'de','do','da','dos','das','em','no','na','nos','nas','um','uma',
    'o','a','os','as','e','ou','que','se','com','por','para','ao',
    'ele','ela','seu','sua','este','essa','isso','mais','como','mas',
    'foi','ser','ter','já','até','só','muito','bem','há','não'
}

def preprocessar(texto: str) -> str:
    return re.sub(r'\s+', ' ', texto.lower().strip())

def criar_explainer(label_names):
    return LimeTextExplainer(class_names=label_names, random_state=42)

def analisar_label(explainer, texto, label_idx, predict_fn,
                   num_features=8, num_samples=300):
    exp = explainer.explain_instance(
        preprocessar(texto), predict_fn,
        labels=[label_idx], num_features=num_features, num_samples=num_samples
    )
    return [(t, p) for t, p in exp.as_list(label=label_idx)
            if t.lower() not in STOPWORDS_PT and len(t) > 2]
