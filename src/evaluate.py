"""evaluate.py — Bootstrap CI and comparative evaluation"""
import numpy as np
import pandas as pd
from functools import partial
from sklearn.utils import resample
from sklearn.metrics import f1_score, hamming_loss


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap: int = 1000):
    """95% bootstrap confidence interval for any metric."""
    scores = []
    n      = len(y_true)
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = resample(range(n), n_samples=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return (float(np.mean(scores)),
            float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


def tabela_comparativa(y_true, modelos_preds: dict, n_bootstrap: int = 1000):
    """Comparative table with Bootstrap CI for multiple models."""
    f1_mac = partial(f1_score, average='macro', zero_division=0)
    f1_mic = partial(f1_score, average='micro', zero_division=0)
    rows   = []
    for nome, preds in modelos_preds.items():
        m_mac,l_mac,u_mac = bootstrap_ci(y_true, preds, f1_mac, n_bootstrap)
        m_mic,l_mic,u_mic = bootstrap_ci(y_true, preds, f1_mic, n_bootstrap)
        m_hl, l_hl, u_hl  = bootstrap_ci(y_true, preds, hamming_loss, n_bootstrap)
        rows.append({'Modelo':nome,
                     'F1_Macro':round(m_mac,4),'F1_Macro_CI':f'[{l_mac:.3f}–{u_mac:.3f}]',
                     'F1_Micro':round(m_mic,4),
                     'Hamming_Loss':round(m_hl,4),'HL_CI':f'[{l_hl:.3f}–{u_hl:.3f}]'})
    return pd.DataFrame(rows)
