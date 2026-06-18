# NurseXAI — Relatório Síntese Final

## Resultado Principal

| Modelo | F1-Macro | IC 95% | Hamming Loss | IC 95% |
|--------|----------|--------|--------------|--------|
| TF-IDF calibrado | 0.527 | [0.456–0.581] | 0.112 | [0.091–0.133] |
| **BERTimbau calibrado ★** | **0.752** | **[0.666–0.825]** | **0.048** | **[0.031–0.064]** |

Calibração de threshold: HL 0.155 → 0.048 (redução de 69%)

## CLIX-M XAI Summary

- ✅ 7/15 labels: Very relevant · Very coherent · High actionability
- ⚠️ 7/15 labels: Relevant · Coherent · Moderate actionability  
- ❌ 1/15 labels: Risco comportamento autolesivo suicida — safety failure

## Limitações Principais

1. Dataset sintético
2. n=49 test set
3. Single LIME explainer (sem ensemble)
4. Single clinical evaluator (sem inter-rater reliability)
5. Sem dados longitudinais
