# Detecção de fraude em cartão de crédito

Pipeline completo: **explorar → preparar → treinar → avaliar → ajustar → explicar**.

Foco: a classe de fraude (1). Com ~0,17% de fraudes, a acurácia engana; olhamos **recall, precisão, F1 e PR-AUC**.
!pip -q install shap imbalanced-learn xgboost
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42
pd.set_option("display.float_format", "{:.4f}".format)
## 1. Explorar
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(URL)
print(df.shape)
df.head()
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))
print("Nulos:", df.isna().sum().sum(), "| Duplicadas:", df.duplicated().sum())
df["Class"].value_counts().plot(kind="bar", title="Distribuição das classes (0 = normal, 1 = fraude)")
plt.show()
# Baseline "burro": prever sempre 'normal'
acc_dummy = (df["Class"] == 0).mean()
print(f"Acurácia prevendo sempre 'não é fraude': {acc_dummy:.4%}  | fraudes detectadas: 0")
# Valor da transação: fraude x normal
print(df.groupby("Class")["Amount"].describe())
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
df["Amount"].hist(bins=60, ax=ax[0]); ax[0].set_title("Amount")
np.log1p(df["Amount"]).hist(bins=60, ax=ax[1]); ax[1].set_title("log1p(Amount)")
plt.show()
## 2. Preparar

- `log1p` no valor (distribuição muito assimétrica).
- Split **estratificado** para manter a proporção de fraudes em treino e teste.
- A padronização fica **dentro do Pipeline**, ajustada só no treino, para não vazar informação do teste.
X = df.drop(columns=["Class"]).copy()
X["Amount_log"] = np.log1p(X["Amount"])
X = X.drop(columns=["Amount"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print("Fraude no treino:", y_train.mean(), "| no teste:", y_test.mean())
print("Fraudes no teste:", y_test.sum())
## 3. Treinar e comparar
def avaliar(nome, modelo, X_te=X_test, y_te=y_test, limiar=0.5):
    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= limiar).astype(int)
    return {
        "modelo": nome, "limiar": limiar,
        "recall": recall_score(y_te, pred),
        "precisão": precision_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "PR-AUC": average_precision_score(y_te, proba),
    }

n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()

modelos = {
    "LogReg (sem peso)": Pipeline([("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "LogReg (balanced)": Pipeline([("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
    "Random Forest (balanced)": Pipeline([("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE))]),
    "XGBoost (scale_pos_weight)": Pipeline([("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              scale_pos_weight=n_neg / n_pos, eval_metric="aucpr",
                              tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE))]),
}

for nome, m in modelos.items():
    m.fit(X_train, y_train)
    print("treinado:", nome)
resultados = pd.DataFrame([avaliar(n, m) for n, m in modelos.items()]).set_index("modelo")
resultados
for nome, m in modelos.items():
    print("=" * 60); print(nome)
    print(classification_report(y_test, m.predict(X_test), digits=4, target_names=["normal", "fraude"]))
## 4. Avaliar: ROC e Precisão x Recall
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for nome, m in modelos.items():
    RocCurveDisplay.from_estimator(m, X_test, y_test, ax=ax[0], name=nome)
    PrecisionRecallDisplay.from_estimator(m, X_test, y_test, ax=ax[1], name=nome)
ax[0].set_title("Curva ROC"); ax[1].set_title("Curva Precisão x Recall (mais informativa aqui)")
plt.tight_layout(); plt.show()
## 5. Ajustar: limiar de decisão
melhor_nome = resultados["PR-AUC"].idxmax()
melhor = modelos[melhor_nome]
print("Melhor PR-AUC:", melhor_nome)

linhas = [avaliar(melhor_nome, melhor, limiar=t) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
tabela_limiar = pd.DataFrame(linhas).set_index("limiar")[["recall", "precisão", "F1"]]
tabela_limiar
tabela_limiar.plot(marker="o", title=f"Trade-off por limiar — {melhor_nome}")
plt.axvline(0.3, ls="--", c="gray"); plt.show()

# Matriz de confusão em 0.5 vs 0.3
from sklearn.metrics import confusion_matrix
proba = melhor.predict_proba(X_test)[:, 1]
for t in (0.5, 0.3):
    tn, fp, fn, tp = confusion_matrix(y_test, proba >= t).ravel()
    print(f"limiar {t}: fraudes pegas={tp}, fraudes perdidas={fn}, falsos alarmes={fp}")
### GridSearchCV escolhendo por recall

Atenção: otimizar **só** recall tende a favorecer modelos que marcam muita coisa como fraude. Por isso também comparo precisão e PR-AUC no teste depois.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

pipe_xgb = Pipeline([("scaler", StandardScaler()),
    ("clf", XGBClassifier(scale_pos_weight=n_neg / n_pos, eval_metric="aucpr",
                          tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE))])
grade = {"clf__max_depth": [3, 5],
         "clf__n_estimators": [100, 200],
         "clf__learning_rate": [0.05, 0.1]}

gs = GridSearchCV(pipe_xgb, grade, scoring="recall", cv=cv, n_jobs=1, verbose=1)
gs.fit(X_train, y_train)
print("Melhores parâmetros:", gs.best_params_)
print("Recall médio (CV):", round(gs.best_score_, 4))
modelo_final = gs.best_estimator_
final = pd.DataFrame([
    avaliar("XGBoost original", modelos["XGBoost (scale_pos_weight)"]),
    avaliar("XGBoost (GridSearch, limiar 0.5)", modelo_final),
    avaliar("XGBoost (GridSearch, limiar 0.3)", modelo_final, limiar=0.3),
]).set_index("modelo")
final
## 6. Explicar: importância das variáveis e SHAP
clf = modelo_final.named_steps["clf"]
imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
imp.sort_values().plot(kind="barh", title="Top 15 — importância (XGBoost)", figsize=(7, 5))
plt.show()
# SHAP em uma amostra do teste (mantém todas as fraudes da amostra para ilustrar)
scaler = modelo_final.named_steps["scaler"]
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

fraudes = X_test_sc[y_test == 1]
normais = X_test_sc[y_test == 0].sample(1500, random_state=RANDOM_STATE)
amostra = pd.concat([fraudes, normais])

explainer = shap.TreeExplainer(clf)
sv = explainer(amostra)

shap.summary_plot(sv, amostra, show=True)
# Por que UMA transação foi marcada como fraude?
idx_fraude = list(amostra.index).index(fraudes.index[0])
shap.plots.waterfall(sv[idx_fraude])
## 7. Extra: undersampling x oversampling

A regra é **testar**. Comparo com regressão logística (rápida), sempre dentro de um pipeline do `imblearn` para o reamostragem acontecer só no treino.
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def pipe_lr(sampler):
    etapas = [("scaler", StandardScaler())]
    if sampler is not None: etapas.append(("sampler", sampler))
    etapas.append(("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)))
    return ImbPipeline(etapas)

variantes = {"LogReg sem reamostragem": pipe_lr(None),
             "LogReg + undersampling": pipe_lr(RandomUnderSampler(random_state=RANDOM_STATE)),
             "LogReg + SMOTE": pipe_lr(SMOTE(random_state=RANDOM_STATE))}
linhas = []
for nome, m in variantes.items():
    m.fit(X_train, y_train); linhas.append(avaliar(nome, m))
pd.DataFrame(linhas).set_index("modelo")
## 8. Conclusões (escreva com os seus números)

- Acurácia prevendo sempre "normal": ...
- Melhor modelo por PR-AUC: ... (recall = ..., precisão = ..., F1 = ...)
- Limiar escolhido: ... e por quê (fraudes perdidas x falsos alarmes): ...
- O que o SHAP mostrou (variáveis que mais pesaram): ...
- O que mudei em relação à Expert: ...
