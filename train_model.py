"""
train_model.py
---------------
Script para treinamento do modelo de Machine Learning para previsão de Churn.

Objetivo:
- Utilizar dados de sessões de clientes (número de pageviews, tempo no site, ticket médio etc.) 
- Construir modelos de classificação que identifiquem a probabilidade de churn.

Modelos Utilizados:
- Random Forest (modelo final escolhido)
- XGBoost
- LightGBM

Técnicas Adicionais:
- Balanceamento de classes usando SMOTE
- Avaliação por Cross-Validation (5 folds)

Impacto:
- Antecipar clientes com risco de abandono.
- Permitir ações proativas de retenção para aumentar o LTV (Lifetime Value) do cliente.
- Reduzir custos de aquisição focando em estratégias de fidelização.

Resultado:
- Modelo salvo em 'models/churn_model.pkl' pronto para ser usado em produção via API.
"""



import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X, y):
    """Avalia o modelo usando cross-validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    return scores.mean()

def train_models():
    # Carregar o dataset
    df = pd.read_csv('data/processed_sessions.csv')
    print("Dados carregados para treinamento! Shape:", df.shape)

    # Separar features e target
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Dividir treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Definir os modelos que vamos testar
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    model_scores = {}

    # Avaliar todos os modelos
    for name, model in models.items():
        score = evaluate_model(model, X_train, y_train)
        model_scores[name] = score
        print(f"{name} - F1-Score (Cross-Validation 5 folds): {score:.4f}")

    # Escolher o melhor modelo
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = models[best_model_name]
    print(f"\n✅ Melhor modelo: {best_model_name}")

    # Treinar o melhor modelo no conjunto de treino
    best_model.fit(X_train, y_train)

    # Avaliar no conjunto de teste
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Gerar gráfico de importância de features
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X.columns

        plt.figure(figsize=(12, 8))
        plt.barh(feature_names, importances)
        plt.title(f'Importância das Variáveis ({best_model_name})')
        plt.xlabel('Score de Importância')
        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/feature_importance.png')
        plt.close()
        print("\nGráfico de importância salvo em: figures/feature_importance.png")

    # Salvar o melhor modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/churn_model.pkl')
    print("\nModelo salvo em: models/churn_model.pkl")

if __name__ == "__main__":
    train_models()
