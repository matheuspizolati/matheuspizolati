# deteccao-fraude-cartao

Detecção de fraude em transações reais de cartão de crédito (base `creditcard.csv`, lida direto pelo link, sem arquivo no repositório).

## 1. O problema
<!-- 99,83% normais x 0,17% fraude. Explique por que a acurácia engana e por que o foco é recall/precisão/F1 da classe 1. Cite: um modelo que diz "normal" para tudo tem ___% de acurácia e 0 fraudes detectadas. -->

## 2. Preparação dos dados
<!-- log1p no Amount; split estratificado (80/20); StandardScaler dentro do Pipeline (só ajusta no treino, evita vazamento). -->

## 3. Comparação dos modelos (classe de fraude)
| Modelo | Recall | Precisão | F1 | PR-AUC |
|---|---|---|---|---|
| LogReg (sem peso) | | | | |
| LogReg (balanced) | | | | |
| Random Forest (balanced) | | | | |
| XGBoost (scale_pos_weight) | | | | |

<!-- Comente qual ficou melhor e por quê. -->

## 4. Limiar de decisão e SHAP
<!-- Limiar escolhido, fraudes perdidas x falsos alarmes. Variáveis que mais pesaram no SHAP. -->

## 5. O que mudei em relação à Expert
<!-- Ex.: scaler no Pipeline para evitar vazamento, PR-AUC como métrica principal, undersampling/SMOTE testados, XGBoost com GridSearch por recall... -->

## Como rodar
Abra `deteccao_fraude_cartao.ipynb` no Google Colab e use **Ambiente de execução > Executar tudo**.
