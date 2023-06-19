# Import library 
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib


# Load model yang sudah dibuat sebelumnya
embedding = joblib.load('embedding.pkl')
retriever = joblib.load('retriever.pkl')

# Import dataset df_combined, hanya kolom ['content', 'title']
df_combined = pd.read_csv('df_combined.csv', usecols=['content', 'title'])

# Fungsi untuk memproses query
def preprocess_query(query):
    # Tokensisasi query
    tokens = word_tokenize(query)
    # Menghapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [w for w in tokens if not w in stop_words]
    # Stemming query
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    # Menggabungkan tokens
    query = ' '.join(tokens)
    return query

# Mendapatkan input dari user
query = input('Masukkan query : ')
query = preprocess_query(query)
print(' > Query akhir :', query)

# Get top document (dokumen dengan nilai cosine similarity tertinggi)
transformed_text = embedding.transform([query])
document_id = retriever.kneighbors(transformed_text, return_distance=False)[0][0]
selected = df_combined.iloc[document_id]['content']

# Print dokumen terpilih (dokumen dengan nilai cosine similarity tertinggi)
print('==============================================================================Hasil pencarian==============================================================================')
print('\nJudul:\n\t', df_combined.iloc[document_id]['title'])
print('\nIsi Dokumen:\n\t', selected)