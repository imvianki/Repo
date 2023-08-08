from fastapi import FastAPI
from functions import get_genre
from functions import get_sentiment
from functions import get_early_access
from functions import get_metascore
from functions import get_specs
from functions import get_games
from prediction import predict_price

app = FastAPI(title='FastAPI by Bianca Torres')

@app.get("/genero/{year}")
def genero(year: str):
    return get_genre(year)

@app.get("/juegos/{year}")
def juegos(year: str):
    return get_games(year)

@app.get("/specs/{year}")
def specs(year: str):
    return get_specs(year)

@app.get("/earlyacces/{year}")
def earlyacces(year: str):
    return get_early_access(year)

@app.get("/sentiment/{year}")
def sentiment(year: str):
    return get_sentiment(year)

@app.get("/metascore/{year}")
def metascore(year: str):
    return get_metascore(year)

@app.get("/prediccion")
def prediccion(genre, year, sentiment, early_access):
    return predict_price(genre, year, sentiment, early_access)








