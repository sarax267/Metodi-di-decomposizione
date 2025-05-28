import numpy as np
from blocchi_gs import blocchi_gs
import time
from loss import compute_loss
from gradient import compute_grad



class GaussSeidel:
    def __init__(self, num_blocks,lambda_reg, tol, max_iter,max_iter_in_block):
        self.lambda_reg = lambda_reg  # Coefficiente di regolarizzazione
        self.tol = tol  # Tolleranza per la convergenza
        self.max_iter = max_iter  # Numero massimo di iterazioni
        self.num_blocks= num_blocks #Numero di Blocchi
        self.execution_time=0
        self.max_iter_in_block=max_iter_in_block


    def gauss_seidel(self, X, y):
        """Metodo di Gauss-Seidel"""

        # Impostazione globale per stampare i vettori con 3 decimali
        np.set_printoptions(precision=3)

        # Inizializzazione dei pesi
        w = np.zeros(X.shape[1])
        loss_history = [] #loss globale, dopo ogni ottimizzazione di blocco
        gradient_history = []
        gradient_norm_history=[]
        loss_history_block = [] #loss globale, dopo ogni ottimizzazione di blocco
        gradient_history_block = []
        gradient_norm_history_block=[]
        w_history=[]
        Time=[]
        iter_count=0 #Voglio contare il numero di iterazioni che il metodo impiega prima di fermarsi per il criterio di arresto
        tot_iter_count=0

        n_features = X.shape[1] # mumero delle colonne di X= numero dello delle features
        block_size = n_features // self.num_blocks #calcolo la dimensione di ogni blocco, e prendo l'intero della divisione
        #print(f"Numero delle features {n_features}")
        #print(f"Numero dei blocchi:{self.num_blocks}")
        #print(f"Dimensione dei blocchi:{block_size}")

        start_time = time.perf_counter()  # Misura il tempo

        w_copy = w.copy()  # Vettore dei pesi corrente
        
        # Calcola la loss iniziale prima di iniziare l'ottimizzazione
        z_initial = X @ w_copy
        initial_loss = compute_loss(X, y, w_copy, z_initial, 0, 0, self.lambda_reg)
        initial_grad = compute_grad(X, y, w_copy, z_initial, 0, 0, self.lambda_reg)
        initial_grad_norm = np.linalg.norm(initial_grad) 
        loss_history.append(initial_loss)
        gradient_history.append(initial_grad)
        gradient_norm_history.append(initial_grad_norm)
        Time.append(0)

        
        for epoch in range(self.max_iter): #Ciclo esterno
           
            for i in range(self.num_blocks):
                #print(f"Sono nel blocco numero {i}")
                start_idx = i * (block_size) #Qui mi dice dove parte un blocco. 
                #print(f"Indice di partenza del blocco {i} è :{start_idx}")
                        #Es i=2, quindi sono nel secondo blocco e la lunghezza di ogni blocco è 3, ho che il secondo blocco parte dalla posizione 3
                        #perchè le posizioni del vettore saranno 0 1 2 | 3 4 5 | 6 7 8 |9 10 11|...
                if i==self.num_blocks-1: # L'ultimo blocco raccoglie i termini in più se ci sono
                    end_idx = start_idx+block_size-1+(n_features % self.num_blocks) #Qui dove finisce un blocco 
                    #print(f"End_index: {end_idx}")
                else: 
                    end_idx = start_idx+block_size-1 #Qui dove finisce un blocco 
                    #print(f"End_index: {end_idx}")
                    #print(f"Start_index: {start_idx}")
                    
                    # Calcola la discesa del gradiente solo su questo blocco
                
                #Ottimizzo il blocco
                loss,grad,grad_norm,w_return,iter_count=blocchi_gs(X,y,w_copy,start_idx,end_idx,self.lambda_reg,self.max_iter,self.tol,self.max_iter_in_block)
                w_copy[start_idx:(end_idx+1)]=w_return[start_idx:(end_idx+1)].copy()

                loss_history_block.append(loss)
                gradient_history_block.append(grad)
                gradient_norm_history_block.append(grad_norm)
                
                tot_iter_count+=iter_count
                #Verifica della compattezza dell'insieme di livello:
                # Controllo della norma di w (limitazione)
                current_norm = np.linalg.norm(w_copy)
                if current_norm > 1e6:
                    raise ValueError("Errore: ||w|| sta divergendo. L'insieme di livello potrebbe non essere limitato.")
            
            #Memorizzo dopo ogni epoca
            z_fin=X@w_copy
            loss_iter=compute_loss(X,y,w_copy,z_fin,0,0,self.lambda_reg)
            loss_history.append(loss_iter)

            w_history.append(w_copy.copy())
            
            grad_iter=compute_grad(X,y,w_copy,z_fin,0,0,self.lambda_reg)
            grad_iter_norm = np.linalg.norm(grad_iter) 
            
            gradient_history.append(grad_iter)
            gradient_norm_history.append(grad_iter_norm)

            # Controllo della loss (chiusura)
            if np.isnan(loss_iter) or np.isinf(loss_iter):
                raise ValueError("Errore: La loss è divergente. L'insieme di livello potrebbe non essere chiuso.")

            end_time_iter=time.perf_counter()
            Time.append(end_time_iter - start_time) #calcolo il tempo di esecuzione per ogni epoca
            
            #Se  non converge, l'itera epoca non converge
            if grad_iter_norm<self.tol:
                break
        
        end_time = time.perf_counter() # Fine del timer
        self.execution_time=end_time - start_time
        
        return loss_history,loss_history_block,gradient_norm_history, gradient_history, epoch, Time 