from sklearn.linear_model import LinearRegression
from collections import Counter
import pandas as pd
import numpy as np
import ast


eda_data = pd.read_csv('Datasets/data_function.csv')

eda_data = eda_data[['genres', 'early_access', 'sentiment', 'year', 'price']]

# Eliminar filas con valores nulos en las columnas seleccionadas
eda_data.dropna(subset=['genres', 'sentiment', 'price'], inplace=True)

values_to_drop = ['Free Demo', 'Play for Free!', 'Install Now', 'Play WARMACHINE: Tactics Demo', 'Free Mod', 'Install Theme', 'Play Now', 'Free HITMAN™ Holiday Pack', 'Play the Demo', 'Free to Try', 'Free to Use', 'Third-party']

val_dict = {value: np.nan for value in values_to_drop}
eda_data.loc[:, 'price'] = eda_data['price'].replace(val_dict)

values_to_modify = ['Free To Play', 'Free to Play', 'Free']
val_dict_modify = {value: 0 for value in values_to_modify}
eda_data.loc[:, 'price'] = eda_data['price'].replace(val_dict_modify)

# Cambia el tipo de valor en la columna 'price'
eda_data['price'] = pd.to_numeric(eda_data['price'], errors='coerce')
print(eda_data['price'].dtype)

values_non_inf = ['1 user reviews', '3 user reviews', '6 user reviews', '5 user reviews', '2 user reviews', '9 user reviews', '8 user reviews', '7 user reviews', '4 user reviews']
non_info_dict = {value: np.nan for value in values_non_inf}
eda_data.loc[:, 'sentiment'] = eda_data['sentiment'].replace(non_info_dict)

eda_data.dropna(subset=['sentiment', 'price'], ignore_index=True, inplace=True)

# Elimina los datos anteriores al 2005
eda_data = eda_data[eda_data['year'] >= 2005]

# Eliminación de precios mayor a 55
eda_data = eda_data.loc[eda_data['price'] <= 55]

# Lista de géneros inútiles para eliminar
useless_genres = ['Animation &amp; Modeling', 'Video Production', 'Software Training', 'Photo Editing', 'Web Publishing', 'Design &amp; Illustration', 'Accounting', 'Utilities', 'Education', 'Audio Production']

# Función para filtrar los géneros útiles de una lista
def filter_useless_genres(genres_list):
    return [genre for genre in genres_list if genre not in useless_genres]

# Aplicar la función de filtrado a la columna 'genres'
eda_data['genres'] = eda_data['genres'].apply(filter_useless_genres)

eda_data['year'] = eda_data['year'].astype(int)


def predict_price(genre, year, sentiment, early_access):
    # Codificar el género como binario
    eda_data['genre_encoded'] = eda_data['genres'].apply(lambda x: genre in x)

    # Convertir el sentimiento en valores numéricos
    sentiment_mapping = {'Mixed': 0, 'Very Positive': 1, 'Positive': 2, 'Mostly Positive': 3,'Mostly Negative': 4, 'Overwhelmingly Positive': 5, 'Negative': 6, 'Very Negative': 7, 'Overwhelmingly Negative': 8}
    sentiment_numeric = sentiment_mapping[sentiment]

    # Crear la columna 'sentiment_numeric' en eda_data
    eda_data['sentiment_numeric'] = eda_data['sentiment'].apply(lambda x: sentiment_mapping[x])

    # Crear el conjunto de características (X)
    X = eda_data[['genre_encoded', 'year', 'early_access', 'sentiment_numeric']]

    # Crear el conjunto de etiquetas (y)
    y = eda_data['price']

    # Crear y ajustar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Crear un diccionario con los valores de las características para la predicción
    input_data = {
        'genre_encoded': genre in eda_data['genres'],
        'year': year,
        'early_access': early_access,
        'sentiment_numeric': sentiment_numeric
    }

    # Crear una fila de datos a partir del diccionario
    input_row = pd.DataFrame([input_data])

    # Realizar la predicción
    y_predict = model.predict(input_row)

    # Calcular el RMSE a partir del MSE
    rmse = np.sqrt(np.mean((y_predict - eda_data['price'])**2))

    result = {'price': y_predict[0], 'RMSE': rmse}
    return result
