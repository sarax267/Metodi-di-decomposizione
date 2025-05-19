import numpy as np
from gradient import compute_grad  # Importa la funzione di gradiente
from gradient import sigmoid
from loss import compute_loss  # Importa la funzione di perdita
import time

# Classe per la regressione logistica con regolarizzazione L2
'''
Per organizzare il codice, possiamo creare una classe 
LogisticRegressionL2, che carica i dati, calcola la loss e il gradiente.
'''
class GradientDescent:
    def __init__(self, lambda_reg, lr, epochs):
        self.lambda_reg =lambda_reg # Regolarizzazione L2
        self.lr = lr  # Tasso di apprendimento
        self.epochs = epochs  # Numero di epoche
        self.w = None #Pesi del modello
        self.execution_time = 0

    def fit(self, X, y):
        n_features = X.shape[1] #mi da il numero delle colonne, cioè delle features 

        # Impostazione globale per stampare i vettori con 3 decimali
        np.set_printoptions(precision=3)

        self.w = np.zeros(n_features)  # Inizializza i pesi a zero
        #print(f"Metodo del Gradiente \n Epoche totali: {self.epochs} \n Tasso di apprendimento: {self.lr}.")
        #print("---------------------------------------------------------------")
        loss_history = []
        gradient_history = []
        gradient_norm_history = []
        start_time = time.perf_counter() # Misura il tempo

        for epoch in range(self.epochs):  # Usa direttamente l'attributo 'epochs'
            grad = self.compute_grad(X, y) # Calcola il gradiente della loss rispetto a w
            self.w -= self.lr * grad  # Aggiornamento dei pesi
            loss = self.compute_loss(X, y)
            loss_history.append(loss)
            gradient_norm_history.append(np.linalg.norm(grad)) #salva le norme dei vettori del gradiente
            gradient_history.append(grad.copy())  # Salva il vettore completo del gradiente

            #if epoch % 50 == 0:
                #print(f"  -> Epoca {epoch}/{self.epochs}\n    - Loss: {loss:.3f}\n    - Gradiente: {grad}")
        
        end_time = time.perf_counter()  # Fine del timer
        self.execution_time=end_time - start_time
        #print(f"\n  -> Gradient Descent tempo esecuzione: {self.execution_time} secondi")
       
        return loss_history, gradient_norm_history, gradient_history, self.w  # Restituisce le loss, i gradienti e i pesi finali


    def predict(self, X):
        return sigmoid(X @ self.w)

    def compute_loss(self, X, y):
        return compute_loss(X,y,self.w,z_block=0,z_before=0,z_after=0,lambda_reg=self.lambda_reg)

    def compute_grad(self, X, y):
        return compute_grad(X,y,self.w,z_block=0,z_before=0,z_after=0,lambda_reg=self.lambda_reg)
