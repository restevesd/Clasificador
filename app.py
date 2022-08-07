#liberías varias
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd # pandas para trabajar arhivos CSV
import time


# importar librerías para la extracción de características 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# pre-procesarmiento de texto
import string
import re

# importar el clasificador desde sklearn
from sklearn.naive_bayes import MultinomialNB


#importar librería para obtener las stopwords y trabajar con textos
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))
check_model = st.empty()

#Descargar Stopwords en español
#Palabras habituales que no aportan significado (stop-words)
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stopword_es = set(nltk.corpus.stopwords.words('spanish'))

#Funciones

#Función para eliminar las tildes
def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

#función para limpiar texto
def limpiar_texto(line_from_column):    
    tokenized_doc = word_tokenize(line_from_column)
    
    new_review = []
    for token in tokenized_doc:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    new_term_vector = []
    for word in new_review:
        if not word in stopword_es:
            new_term_vector.append(normalize(word.lower()))
    
    final_doc = []
    for word in new_term_vector:
        final_doc.append(wordnet.lemmatize(word))
    
    return ' '.join(final_doc)

pickled_model = pickle.load(open('model.pkl', 'rb'))
st.title ("Aplicación para detectar posibles Fake News")
st.header("Por favor ingrese el texto de la noticia para verificar si es falsa o verdadera")

texto = st.text_area('Noticia', height=300)
bandera = 0

if st.button('Analizar !!!!'):
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        vect = pickle.load(open('carecteristicas.pkl', 'rb'))
        
        check_model = st.success("Modelo de clasificación cargado correctamente")
        time.sleep(1) # Sleep for 3 seconds
        check_model.empty()
        bandera = 1
    except:
        st.error('No se pudo cargar el modelo de detección')

if bandera == 1:
    texto_df = pd.DataFrame({"texto":texto},index=range(1))
    texto_df['clean'] = texto_df['texto'].apply(limpiar_texto)
    test_dtm = vect.transform(texto_df["clean"]) # use it to extract features from training data
    y_pred_test = model.predict(test_dtm) # make class predictions for test_dtm
    y_pred_proba = model.predict_proba(test_dtm).tolist()
    resul = np.argmax(y_pred_proba)
    resul = round(y_pred_proba[0][resul]  * 100,2)
    print(resul)

    
    # Convertir etiquetas Fake - Real
    texto_df['class'] = pd.Series(y_pred_test).map({1:"Real", 0:"Falso"}) # Fake is 1, Not Fake is 0. 
    
    if y_pred_test == "0":
        html_str = f"""<h3><span style='color:red'>Esta noticia parece falsa !! con una probabilidad del {resul}% </span></h3>"""
        st.markdown(html_str, unsafe_allow_html=True)
    else:
        html_str = f"""<h3><span style='color:green'>Esta noticia parece verdadera !! con una probabilidad del {resul}% </span></h3>"""
        st.markdown(html_str, unsafe_allow_html=True)