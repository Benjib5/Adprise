import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


df = pd.read_csv('social_media_1.csv')
for i in range(9):
    df = pd.concat([df, pd.read_csv(f'social_media_{i+2}.csv')])
    
# Colunas categóricas
categorical_columns = ['categoria', 'genero', 'nacional']
interest_columns = ['Saúde e bem-estar',
       'Educação e aprendizado', 'Esportes', 'Fotografia', 'Fitness',
       'Carros e automóveis', 'Finanças e investimentos',
       'Atividades ao ar livre', 'Parentalidade e família', 'História',
       'Jogos', 'Música', 'Tecnologia', 'Moda', 'Faça você mesmo e artesanato',
       'Livros', 'Negócios e empreendedorismo', 'Natureza', 'Beleza',
       'Ciência', 'Alimentos e refeições', 'Causas sociais e ativismo',
       'Jardinagem', 'Filmes', 'Arte', 'Culinária', 'Viagem', 'Política',
       'Animais de estimação']  # Encontrar as colunas de interesse

# Definindo o transformer para as variáveis categóricas e as colunas de interesse
# Usaremos o OneHotEncoder para as variáveis categóricas e interesses
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),  # Categóricas (OneHotEncoding)
        ('interest', 'passthrough', interest_columns),  # Manter as colunas de interesse inalteradas
        ('age', StandardScaler(), ['idade'])  # Escalonamento da variável idade
    ])

# Preparando o modelo KNN dentro de um Pipeline
knn = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Pré-processamento
    ('classifier', KNeighborsClassifier(n_neighbors=11))  # Classificador KNN
])

# Separando as variáveis independentes (X) e a variável dependente (y)
X = df.drop(columns=['Unnamed: 0','rede social', 'pais'])
y = df['rede social']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo KNN
knn.fit(X_train, y_train)

joblib.dump(knn, "knn_treinado.pkl")
