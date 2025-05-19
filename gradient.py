import numpy as np

# Funzione sigmoide sigma(z)=1/(1+e^(-z)) dove considero y={0,1}
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Funzione per calcolare il gradiente
'''
Ritorna:
    - grad: vettore di dimensione (n_features,) contenente il gradiente
'''


def compute_grad(X,y,w,z_block,z_before,z_after,lambda_reg):
    n = X.shape[0]
    if ((np.all(z_block==0)) and (np.all(z_before==0)) and (np.all(z_after==0))):
        z_total=X@w
    else:
        z_total = z_block + z_before + z_after
    r = -y * sigmoid(-y*z_total) 
    #print(f"Dimensioni r: {r.shape[0]}") #ok perchè dim(r)=n=100
    #print(f"Dim X trasposto : {(X.T).shape[0]}")
    grad = (X.T @ r) / n + lambda_reg * w
    #print(f"Dim gradiente di grad: {grad.shape[0]}")
    return grad




