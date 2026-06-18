# NurseXAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/[ZENODO_DOI].svg)](https://doi.org/[ZENODO_DOI])

Multi-label classification of NANDA-I nursing diagnoses from synthetic psychiatric clinical notes in European Portuguese (PT-PT) using BERTimbau with per-label threshold calibration and LIME explainability evaluated against the CLIX-M framework.

## Resultados Principais

| Modelo | F1-Macro | IC 95% | Hamming Loss | IC 95% |
|--------|----------|--------|--------------|--------|
| TF-IDF (calibrado) | 0·527 | [0·456–0·581] | 0·112 | [0·091–0·133] |
| **BERTimbau (calibrado) ★** | **0·752** | **[0·666–0·825]** | **0·048** | **[0·031–0·064]** |

IC 95% por bootstrap (n=1000) · holdout fixo n=49

## Estrutura

```
nursexai/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_lime_clix_analysis.ipynb
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── lime_analysis.py
├── data/
│   ├── dataset_info.json
│   └── split_indices.json
├── outputs/
│   ├── metrics_table.csv
│   ├── per_label_results.csv
│   ├── thresholds_kfold.csv
│   └── lime_tokens.csv
├── report/final_report.md
├── references/references.md
├── LICENSE
├── LICENSE-DATA
├── CITATION.cff
└── requirements.txt
```

## Como Usar

```bash
# 1. Clonar
git clone https://github.com/cicerosiq/nursexai.git
cd nursexai

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar notebooks em ordem no Google Colab (GPU T4)
# Fazer upload do dataset em data/notas_dataset_completo.csv
```

## Citação

```bibtex
@misc{nursexai2025,
  title  = {NurseXAI: Multi-label NANDA-I nursing diagnosis classification in European Portuguese},
  author = {Cicero Mendes Siqueira },
  year   = {2025},
  doi    = {[ZENODO_DOI]},
  url    = {https://github.com/Cicerosiq/nursexai}
}
```

## Referências Principais

- Souza et al. (2020). BERTimbau. BRACIS.
- Brankovic et al. (2025). CLIX-M. npj Digital Medicine, 8:364.
- NANDA International (2024). Nursing Diagnoses 2024–2026. Thieme.
