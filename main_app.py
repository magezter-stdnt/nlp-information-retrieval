import re
import tkinter as tk
from tkinter import messagebox
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

# Fungsi untuk meng-handle event saat tombol "Cari" ditekan
def search_query():
    # Mendapatkan input dari user
    query = entry_query.get()
    
    # Validasi input query
    if not query:
        messagebox.showerror("Error", "Input query tidak boleh kosong!")
        return
    # if not re.match("^[a-zA-Z0-9\s]+$", query):
    #     messagebox.showerror("Error", "Input query hanya boleh berisi huruf, angka, dan spasi!")
    #     return
    
    query = preprocess_query(query)
    label_query.config(text="Query akhir: " + query)
    
    # Get top document (dokumen dengan nilai cosine similarity tertinggi)
    transformed_text = embedding.transform([query])
    document_id = retriever.kneighbors(transformed_text, return_distance=False)[0][0]
    selected = df_combined.iloc[document_id]['content']
    
    # Menampilkan dokumen terpilih (dokumen dengan nilai cosine similarity tertinggi)
    messagebox.showinfo("Hasil Pencarian", "Judul:\n" + df_combined.iloc[document_id]['title'] + "\n\nIsi Dokumen:\n" + selected)

# Membuat window GUI (akan berisikan label, entry, button)
window = tk.Tk()
window.title("Sistem Information Retrieval")
window.geometry("400x200")

frame = tk.Frame(window)
frame.pack(expand=True, pady=50)

# Membuat komponen GUI (label, entry, button)
label_query = tk.Label(frame, text="Masukkan query:")
label_query.pack()

entry_query = tk.Entry(frame)
entry_query.pack()
entry_query.configure(width=50)

button_search = tk.Button(frame, text="Cari", command=search_query)
button_search.pack()

# Menjalankan window GUI dengan event loop (agar window tidak tertutup saat program dijalankan)
window.mainloop()
