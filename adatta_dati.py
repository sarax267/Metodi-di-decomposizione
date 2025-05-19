from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
def calcolo(dataset):
    X = dataset.data  # Features
    #Normalizzo i dati
    X = StandardScaler().fit_transform(X)
    y_base = dataset.target  
    Campioni=X.shape[0]
    Caratteristiche=X.shape[1]
    return X,y_base,Campioni,Caratteristiche

def df_sostituisco_target(dataset,y):
    # creiamo una variabile "data" che contiene il dataframe dal dataset
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data['target'] = pd.Series(dataset['target'], dtype='category')
    #Cambio i valori del dataframe del target con y
    data['target'] = y
    return data

def estrai_dati(name_dataset,dataset):

    if(name_dataset=="Fetch"):
        
        # I dati sono testi, quindi dobbiamo convertirli in feature numeriche usando TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000)  # Limita a 1000 feature per semplicità
        X= vectorizer.fit_transform(dataset.data).toarray()
        y_base=dataset.target
        y = np.where(y_base == 0, -1, y_base)  # Labels (-1 = maligno, 1 = benigno)
        # In digits non abbiamo feature_names, quindi creiamo nomi artificiali
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        data = pd.DataFrame(data=X, columns=feature_names)
        data['target'] = y
        Campioni = X.shape[0]
        Caratteristiche = X.shape[1]

    if(name_dataset=="Breast_Cancer"):
        X,y_base,Campioni,Caratteristiche=calcolo(dataset)
        # Labels (0 = maligno, 1 = benigno)
        # Trasforma 0 in -1 e lascia 1 invariato
        y = np.where(y_base == 0, -1, y_base) # Labels (-1 = maligno, 1 = benigno)
        data=df_sostituisco_target(dataset,y)

    if(name_dataset=="Diabetes"):
        X,y_base,Campioni,Caratteristiche=calcolo(dataset)
        # Trasforma il target in binario (0/1) usando la mediana dei target come soglia
        y1 = (y_base > np.median(y_base)).astype(int)
        y = np.where(y1 == 0, -1, y1)  # Labels (-1 = maligno, 1 = benigno)
        data=df_sostituisco_target(dataset,y)

    if(name_dataset=="Digits"):
        X,y_base,Campioni,Caratteristiche=calcolo(dataset)
        #nella colonna dei target ci sono i numeri da 0 a 9 e li trasformo in 0 se pari e 1 se dispari
        y1=(y_base % 2).astype(int)
        y = np.where(y1 == 0, -1, y1)  # Labels (-1 = maligno, 1 = benigno)
        data=df_sostituisco_target(dataset,y)

    if(name_dataset=="Iris"):
        X,y_base,Campioni,Caratteristiche=calcolo(dataset)
        #0: Iris Setosa, 1: Iris Versicolor, 2: Iris Virginica
        y1=(y_base== 0).astype(int)  # 1 per Setosa, 0 per le altre
        y = np.where(y1 == 0, -1, y1)  # Labels (-1 = maligno, 1 = benigno)
        data=df_sostituisco_target(dataset,y)

    if(name_dataset=="Wine"):
        X,y_base,Campioni,Caratteristiche=calcolo(dataset)
        #0: vino A, 1: vino B, 2: vino C
        y1=(y_base== 0).astype(int)  # 1 per vino A, 0 per le altre
        y = np.where(y1 == 0, -1, y1)  # Labels (-1 = maligno, 1 = benigno)
        data=df_sostituisco_target(dataset,y)
    
    if(name_dataset=="Adult"):
    
                # Converte il Bunch in un DataFrame
        data = dataset.frame.copy()  # solo se fetch_openml(..., as_frame=True)

        # Estrai e trasforma il target
        y = data["class"].apply(lambda val: -1 if val == "<=50K" else 1).astype(int).to_numpy()

        # Rimuove la colonna target per ottenere solo le feature
        X = data.drop(columns="class")

        # One-hot encoding delle variabili categoriche
        X = pd.get_dummies(X)
        column_names = X.columns  # salva i nomi originali

        # Normalizzazione
        X = StandardScaler().fit_transform(X)

        # Ricostruisce un nuovo DataFrame con nomi originali
        data = pd.DataFrame(X, columns=column_names)
        data["target"] = y

        # ✅ Seleziona solo le prime 500 righe
        data = data.head(500)

        # Aggiorna le variabili da restituire
        X = data.drop(columns="target").to_numpy()
        y = data["target"].to_numpy()
        Campioni = X.shape[0]
        Caratteristiche = X.shape[1]


    

    return X,y,Campioni,Caratteristiche,data
