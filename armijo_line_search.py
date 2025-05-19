import numpy as np
from gradient import compute_grad
#from tqdm import tqdm

class ArmijoLineSearch:
    def __init__(self, alpha_init=1, beta=0.5, sigma=1e-4,lambda_reg=0.1):
        """
        Inizializza i parametri di Armijo Line Search.
        """
        self.alpha_init = alpha_init
        self.beta = beta
        self.sigma = sigma
        self.lambda_reg=lambda_reg

    # In armijo_line_search.py:
    def search(self, f, grad, X, y, w, direction):
        alpha = self.alpha_init
        original_loss = f(w)  # f ora riceve solo w
        
        while True:
            w_new = w + alpha * direction
            try:
                new_loss = f(w_new)
                expected_reduction = self.sigma * alpha * np.dot(grad, direction) #costante*passo*gradiente trasposto direzione
                #print(f"Armijo check: {new_loss} <= {original_loss + expected_reduction:}?")
                if new_loss <= original_loss + expected_reduction:
                    #print("Si!Riduco alpfa?No")
                    break
                #print("No!Riduco alpha?Si")
                #print("alpha:",alpha)
                alpha *= self.beta
                if alpha < 1e-10:
                    break
            except ValueError as e:
                print(f"Error in Armijo search: {e}")
                alpha *= self.beta
                continue
        #print("alpha:",alpha)
        return alpha

    def compute_grad(self, X, y,w):
        return compute_grad(X,y,w,z_block=0,z_before=0,z_after=0,lambda_reg=self.lambda_reg)