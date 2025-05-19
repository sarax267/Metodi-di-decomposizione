#mi serve numpy per i calcoli con matrice
import numpy as np

# Funzione di perdita con regolarizzazione L2
"""
    Calcola la funzione di perdita logistica regolarizzata con L2.
    
    Parametri:
    - X: matrice dei dati (n_samples, n_features)
    - y: etichette (-1 o 1), vettore di dimensione (n_samples,)
    - w: pesi del modello, vettore di dimensione (n_features,)
    - reg_lambda: coefficiente di regolarizzazione
    
    Ritorna:
    - loss: valore scalare della funzione di perdita
"""

import numpy as np
#logistic
def compute_loss(X,y,w,z_block,z_before,z_after,lambda_reg):
    n = X.shape[0]
    if ((np.all(z_block==0)) and (np.all(z_before==0)) and (np.all(z_after==0))):
        z_total=X@w
    else:
        z_total = z_block + z_before + z_after
    loss = np.mean(np.log(1 + np.exp(-y * z_total)))
    # Regularizzazione su TUTTI i pesi (w contiene tutti i pesi)
    reg_term = (lambda_reg / 2) * np.linalg.norm(w) ** 2
    return loss+reg_term



