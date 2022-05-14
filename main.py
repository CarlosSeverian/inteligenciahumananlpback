from flask import Flask, request, jsonify, json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.naive_bayes import MultinomialNB
import joblib
import re
import unicodedata

app = Flask(__name__)

modelo = joblib.load('modelos/modelo.pkl')
vetorizador = joblib.load('modelos/vetorizador.pkl')
stw = stopwords.words('portuguese')
pontuacao = list(punctuation)
stw_pt = set(stw + pontuacao)

@app.route("/")
def root():
    resp = {}
    resp['status'] = "funcionando"
    return jsonify( resp ), 200

@app.route("/predict/",  methods=['GET', 'POST'])
def predict():

    # capturar o json enviado
    dados = request.get_json()
    texto_usuario = dados['avaliacao']

    # tratamento do texto recebido
    frase = word_tokenize(texto_usuario.lower())
    frase = pipeline(frase, stw_pt)
    frase_lista = [frase]
    
    # Vetorizador pré-treinado prepara a frase para envio ao modelo
    Bow_frase = vetorizador.transform( frase_lista ) 

    # modelo pré-treinado classifica a frase que vem do vetorizador
    saida = modelo.predict(Bow_frase)
    dict_saida = {0:"negativa 😞", 1:"positiva 😃"}
    Predicao = dict_saida[saida[0]]
    
    # devolve o retorno em formato json para o processo frontend solicitante
    resp = {}
    resp['avaliacao'] = dict_saida[saida[0]]
    return jsonify( resp ), 200

def pipeline(i, stw_pt):
    frase = no_alphas(i)
    frase = remove_stops(frase, stw_pt )
    frase = remove_acento(frase)
    frase = remontar_frase(frase)

    return frase


def no_alphas( frase_tokenizada ):
  return [palavra for palavra in frase_tokenizada if (palavra.isalpha())]

# Remove stop words
def remove_stops( frase_tokenizada, stopwords_pt ):
  w_token_1_sem_stopwords = [palavra for palavra in frase_tokenizada if palavra not in stopwords_pt]
  return w_token_1_sem_stopwords

# Remove emojis
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# ============= Cuidado ao usar, ela é lenta! ===============
def removerAcentosECaracteresEspeciais(palavra):
    # Unicode normalize transforma um caracter em seu equivalente em latin.
    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
    return re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)
# ============= Cuidado ao usar, ela é lenta! ===============

# Remove acentos
def remove_acento( frase_tokenizada ):
  frase = []
  for palavra in frase_tokenizada:
    palavra = palavra.replace('á','a')
    palavra = palavra.replace('é','e')
    palavra = palavra.replace('í','i')
    palavra = palavra.replace('ó','o')
    palavra = palavra.replace('ú','u')
    palavra = palavra.replace('ä','a')
    palavra = palavra.replace('ë','e')
    palavra = palavra.replace('ï','i')
    palavra = palavra.replace('ö','o')
    palavra = palavra.replace('ü','u')
    palavra = palavra.replace('ã','a')
    palavra = palavra.replace('õ','o')
    palavra = palavra.replace('ç','c')
    palavra = palavra.replace('â','a')
    palavra = palavra.replace('ê','e')
    palavra = palavra.replace('î','i')
    palavra = palavra.replace('ô','o')
    palavra = palavra.replace('û','u')
    palavra = palavra.replace('à','a')
    palavra = palavra.replace('è','e')
    palavra = palavra.replace('ì','i')
    palavra = palavra.replace('ò','o')
    palavra = palavra.replace('ù','u')
    frase.append( palavra )
  return frase


# Transformar frase tokenizada em frase novamente
def remontar_frase(frase_tokenizada):
  return " ".join(frase_tokenizada)



if __name__ == "__main__":
    app.run(port=5000,host='0.0.0.0', debug=False)