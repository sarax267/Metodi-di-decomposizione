import numpy as np
from gradient import compute_grad  # Importa la funzione di gradiente
from loss import compute_loss  # Importa la funzione di perdita
from armijo_line_search import ArmijoLineSearch
from gradient import sigmoid
#from tqdm import tqdm  # Importa la libreria per la barra di progresso
import time

# Classe per Gradient Descent Armijo
class GradientDescentArmijo:
    def __init__(self, lambda_reg, epochs, armijo_params=None):
        """
        Inizializza il modello con regolarizzazione L2 e ricerca di Armijo.
        """
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.w = None
        self.execution_time=0

        # Creazione dell'oggetto ArmijoLineSearch con parametri personalizzabili
        self.armijo = ArmijoLineSearch(**(armijo_params or {}))


    def gradient_descent_armijo(self, X, y):
        """
        Esegue la discesa del gradiente con ricerca della linea di Armijo.
        """
        loss_history = []
        gradient_history = []
        gradient_norm_history = []
        self.w = np.zeros(X.shape[1])  # Inizializza i pesi qui
        start_time = time.perf_counter() # Misura il tempo

        for epoch in range(self.epochs):
            grad = self.compute_grad(X, y, self.w)
            direction = -grad  # Direzione di discesa
            loss=self.compute_loss(X,y,self.w)
            # Usa la classe Armijo per trovare il miglior alpha
            alpha = self.armijo.search(
                lambda w_new: self.compute_loss(X, y, w_new),
                grad, X, y, self.w, direction
            )
            #print("Alfa gda",alpha)
            # Aggiorna i pesi
            self.w = self.w + alpha * direction

            #Memorizzo Loss e Gradient
            loss_history.append(loss)
            gradient_norm_history.append(np.linalg.norm(grad))
            gradient_history.append(grad.copy())  # Salva il vettore completo del gradiente
            #if epoch % 50 == 0:
                #print(f"  -> Epoca {epoch}/{self.epochs}\n    - Loss: {loss:.3f}\n    - Gradiente: {grad}")

        end_time = time.perf_counter()  # Fine del timer
        self.execution_time=end_time - start_time
        #print(f"\n -> Gradient Descent Armijo Tempo: {self.execution_time} secondi")
        return loss_history,gradient_norm_history, gradient_history, self.w  # Restituisce le loss, i gradienti e i pesi finali

    def fit(self, X, y):
        """
        Addestra il modello.
        """
        #print(f"Metodo del Gradient Descent Armijo \n Epoche totali:{self.epochs}") 
        #print("---------------------------------------------------------------")

        # Impostazione globale per stampare i vettori con 3 decimali
        np.set_printoptions(precision=3)

        self.w = np.zeros(X.shape[1])  # Inizializza i pesi
        return self.gradient_descent_armijo(X, y)

    def predict(self, X):
        """
        Calcola le probabilità previste.
        """
        return sigmoid(X @ self.w)

    # In gradient_descent_armijo.py, modifica:
    # In gradient_descent_armijo.py, modifica compute_loss:
    def compute_loss(self, X, y, w):
        return compute_loss(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)

    def compute_grad(self, X, y, w):
        return compute_grad(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)
    
    