"""
Projeto: Predição do Nível de Liquidez em Fundos de Investimento

Integrantes:
- Bruna Sousa VAlerio dos Santos - 10418640 - 10418640@mackenzista.com.br
- Fernanda de Moraes Brazolin - 10417732 - 10417732@mackenzista.com.br
- Danilo Ferreira Rocha - 10402374 - 10402374@mackenzista.com.br
- Yasmin Reis Toledo - 10419669 - 10419669@mackenzista.com.br


Descrição:
Este arquivo contém o código de análise exploratória e modelagem de Machine Learning
para classificação do nível de liquidez de fundos de investimento com base em dados da ANBIMA.

"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# CARREGAR BASE
arquivo = "FUNDOS-175-CARACTERISTICAS-PUBLICO.xlsx"

try:
    df = pd.read_excel(arquivo)
    print("Base carregada com sucesso.")
except FileNotFoundError:
    print(f"Erro: o arquivo '{arquivo}' não foi encontrado na pasta do projeto.")
    raise

# SELECIONAR COLUNAS RELEVANTES
colunas_utilizadas = [
    "Categoria ANBIMA",
    "Tipo ANBIMA",
    "Composição do Fundo",
    "Aberto Estatutariamente",
    "Fundo ESG",
    "Tributação Alvo",
    "Tipo de Investidor",
    "Característica do Investidor",
    "Adaptado 175",
    "Foco Atuação",
    "Nível 1 Categoria",
    "Nível 2 Categoria",
    "Nível 3 Subcategoria",
    "Prazo Pagamento Resgate em dias"
]

colunas_existentes = [col for col in colunas_utilizadas if col in df.columns]
colunas_faltantes = [col for col in colunas_utilizadas if col not in df.columns]

if colunas_faltantes:
    print("\nAviso: as seguintes colunas não foram encontradas na base:")
    for col in colunas_faltantes:
        print("-", col)

df = df[colunas_existentes].copy()

print("\nColunas utilizadas no projeto:")
for col in df.columns:
    print("-", col)

# 3. TRATAMENTO INICIAL
df = df.dropna(subset=["Prazo Pagamento Resgate em dias"]).copy()

df["Prazo Pagamento Resgate em dias"] = pd.to_numeric(
    df["Prazo Pagamento Resgate em dias"],
    errors="coerce"
)

df = df.dropna(subset=["Prazo Pagamento Resgate em dias"]).copy()

#  CRIA VARIÁVEL-ALVO: NÍVEL DE LIQUIDEZ
def classificar_liquidez(prazo):
    if prazo <= 4:
        return "Alta"
    elif prazo <= 30:
        return "Média"
    else:
        return "Baixa"

df["Nível de Liquidez"] = df["Prazo Pagamento Resgate em dias"].apply(classificar_liquidez)

print("\nDistribuição da variável-alvo:")
print(df["Nível de Liquidez"].value_counts())

# DEFINI X (ENTRADAS) E y (ALVO)
X = df.drop(columns=["Prazo Pagamento Resgate em dias", "Nível de Liquidez"])
y = df["Nível de Liquidez"]

colunas_categoricas = X.columns.tolist()

preprocessador = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            colunas_categoricas
        )
    ]
)

#  MODELO DE MACHINE LEARNING

modelo = Pipeline(steps=[
    ("preprocessamento", preprocessador),
    ("classificador", RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

#  SEPARA TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamanho do conjunto de treino:", X_train.shape[0])
print("Tamanho do conjunto de teste:", X_test.shape[0])

#  TREINAR O MODELO
modelo.fit(X_train, y_train)

# FAZER PREVISÕES
y_pred = modelo.predict(X_test)

# AVALIAR O MODELO
acuracia = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("RESULTADOS DO MODELO")
print("=" * 60)
print(f"Acurácia: {acuracia:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# EXIBIR MATRIZ DE CONFUSÃO
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de Confusão - Classificação do Nível de Liquidez")
plt.tight_layout()
plt.show()

# IMPORTÂNCIA DAS VARIÁVEIS
onehot = modelo.named_steps["preprocessamento"].named_transformers_["cat"].named_steps["onehot"]
nomes_features = onehot.get_feature_names_out(colunas_categoricas)

importancias = modelo.named_steps["classificador"].feature_importances_

df_importancias = pd.DataFrame({
    "Variável": nomes_features,
    "Importância": importancias
}).sort_values(by="Importância", ascending=False)

print("\nTop 15 variáveis mais importantes:")
print(df_importancias.head(15))

top15 = df_importancias.head(15).copy()

plt.figure(figsize=(10, 6))
plt.barh(top15["Variável"][::-1], top15["Importância"][::-1])
plt.title("Top 15 Variáveis Mais Importantes no Modelo")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()
