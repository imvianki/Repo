from datetime import datetime
import pandas as pd
import numpy as np
import ast


# Importar datos y convertirlos a un dataset

with open('Datasets/steam_games.json', 'r') as file:
    data_list = [ast.literal_eval(record) for record in file.readlines()]

data = pd.DataFrame(data_list)

# filtrado de datos que no son de formato "YYYY-MM-DD"
data[pd.to_datetime(data['release_date'], format='%Y-%m-%d', errors='coerce').isna()]

# Devuelve un formato estándar

def get_format(date):
    formats_list = ['%Y-%m-%d', '%d.%m.%Y', '%b %Y', '%b-%y', '%B %Y', '%Y', '%d %b, %Y']
    for formats in formats_list:
        try:
            parsed_date = datetime.strptime(date, formats)
            if parsed_date.strftime(formats) == date:
                return parsed_date.date().strftime('%Y-%m-%d')
        except ValueError:
            continue
        except TypeError:
            return np.nan
    return np.nan

data['modify_date'] = data['release_date'].apply(get_format)

# Eliminamos los valores NaN de la columna 'modify_date'
data.dropna(subset=['modify_date'], inplace=True)

data['modify_date'] = pd.to_datetime(data['modify_date'])
data['year'] = data['modify_date'].dt.year

# Convierte los datos de year a string
data['year'] = data['year'].astype(str)

# Guardar archivo limpio
data.to_csv('Datasets/data_function.csv')


# Function N°1

def get_genre(year):
    filtered_genre = data[data['year'] == str(year)]
    genres = [genre for lista in filtered_genre['genres'] if isinstance(lista, list) for genre in lista if pd.notna(genre)]
    genres_dict = {genre: genres.count(genre) for genre in genres}
    top_genres = dict(sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    return top_genres

# Function N°2

def get_games(year):
    filtered_games = data[data['year'] == str(year)]
    games_list = [game for game in filtered_games['app_name'].unique() if pd.notna(game)]
    games_dict = {year: games_list}
    return games_dict

# Function N°3

def get_specs(year):
    filtered_specs = data[data['year'] == str(year)]
    specs = [spec for lista in filtered_specs['specs'] if isinstance(lista, list) for spec in lista if pd.notna(spec)]
    specs_dict = {spec: specs.count(spec) for spec in specs}
    top_specs = dict(sorted(specs_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    return top_specs

# Function N°4

def get_early_access(year):
    filtered_early_access = data[data['year'] == str(year)]
    early_access_count = filtered_early_access[filtered_early_access['early_access'] == True].shape[0]
    early_access_dict = {year: early_access_count}
    return early_access_dict

# Function N°5

# Según el año de lanzamiento, se devuelve un diccionario con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.

def get_sentiment(year):
    sentiments = ['Mixed', 'Mostly Negative', 'Mostly Positive', 'Negative', 'Overwhelmingly Negative',
                  'Overwhelmingly Positive', 'Positive', 'Very Negative', 'Very Positive']
    filtered_sentiment = data[data['year'] == str(year)]
    filtered_sentiment = filtered_sentiment[filtered_sentiment['sentiment'].isin(sentiments)]
    sentiment_list = [x for x in filtered_sentiment['sentiment'] if pd.notna(x)]
    sentiment_dict = {sentiment: sentiment_list.count(sentiment) for sentiment in sentiment_list}
    return dict(sorted(sentiment_dict.items(), key=lambda x: x[1], reverse=True))

# Function N°6

def get_metascore(year):
    filtered_metascore = data[(data['year'] == str(year)) & (data['metascore'] != 'NA')]
    filtered_metascore = filtered_metascore.dropna(subset=['metascore'])
    sorted_metascore = filtered_metascore.sort_values('metascore', ascending=False)
    top_data = sorted_metascore.head(5)[['app_name', 'metascore']]
    if top_data.empty:
        metascore_dict = {'Data not found'}
    else:
        metascore_dict = top_data.set_index('app_name')['metascore'].to_dict()
    return metascore_dict