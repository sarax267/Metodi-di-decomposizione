import numpy as np
from gradient import compute_grad
from loss import  compute_loss
from blocchi_j import blocchi_j
import time
from joblib import Parallel, delayed
#creo una memoria condivisa per w
from multiprocessing.shared_memory import SharedMemory 


class Jacobi:
    def __init__(self, num_blocks, lambda_reg, tol, max_iter, num_jobs):
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.max_iter = max_iter
        self.num_blocks = num_blocks
        self.num_jobs = num_jobs
        self.execution_time = 0


    def jacobi(self, X, y):

        # Impostazione globale per stampare i vettori con 3 decimali
        np.set_printoptions(precision=3)

        n_features = X.shape[1]

        # Creiamo una memoria condivisa per w
        shm = SharedMemory(create=True, size=n_features * np.float64().itemsize)
        w= np.ndarray(n_features, dtype=np.float64, buffer=shm.buf)
        w.fill(0)  # Inizializziamo w a zero

        loss_history = []
        gradient_history = []
        gradient_norm_history = []
        w_history=[]
        iter_history=np.zeros(self.num_blocks)
        iter_count = 0

        
        block_size = n_features // self.num_blocks

        start_time = time.perf_counter()

        for iteration in range(self.max_iter):
            w_old = w.copy()
            iter_count += 1

            def update_block(i, w_current):
                #print(f"Sono nel blocco numero {i+1}")
                start_idx = i * (block_size) #Qui mi dice dove parte un blocco. 
                #print(f"Indice di partenza del blocco {i} è :{start_idx}")
                    #Es i=2, quindi sono nel secondo blocco e la lunghezza di ogni blocco è 3, ho che il secondo blocco parte dalla posizione 3
                    #perchè le posizioni del vettore saranno 0 1 2 | 3 4 5 | 6 7 8 |9 10 11|...
                if i==self.num_blocks-1: # L'ultimo blocco raccoglie i termini in più se ci sono
                    end_idx = start_idx+block_size-1+(n_features % self.num_blocks) #Qui dove finisce un blocco 
                    
                else: 
                    end_idx = start_idx+block_size-1 #Qui dove finisce un blocco
                
                try:
                    # Chiamata corretta a bloccho_loss_grad_pesi con tutti i parametri
                    loss, grad,grad_norm, updated_w,iter = blocchi_j(X,y,w,start_idx,end_idx,self.lambda_reg,self.max_iter,self.tol)
                    #print(f"Update_block: {updated_w[start_idx:(end_idx+1)]}")
                    

                    return start_idx,end_idx,updated_w[start_idx:(end_idx+1)],i,iter
                except Exception as e:
                    print(f"Error in block {i+1}: {e}")
                    return start_idx,end_idx,w_current[start_idx:(end_idx+1)],i,iter

            # Parallelizzazione
            results = Parallel(n_jobs=self.num_jobs)(
                delayed(update_block)(i, w.copy()) for i in range(self.num_blocks)
            )
            
            #print(f"risultati: {results}")
            # Aggiornamento pesi nella memoria condivisa
            for res in results:
                if res is not None:
                    start_idx, end_idx, w_block,i_block,iterazione_block = res
        
                if len(w_block) > 0:  # Solo se il blocco non è vuoto
                    w[start_idx:(end_idx+1)] = w_block
                    iter_history[i_block] += iterazione_block
            
            # Calcolo loss e gradiente globali
            loss = compute_loss(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)
            grad = compute_grad(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)
            
            grad_norm=np.linalg.norm(grad)

            #Appendo nella lista
            loss_history.append(loss)
            gradient_history.append(grad)
            gradient_norm_history.append(grad_norm)
            w_history.append(w)
            
            if np.linalg.norm(w - w_old) < self.tol:
                break
        #Quante volte ogni blocco viene itinerato
        #print(iter_history)
        end_time = time.perf_counter()
        self.execution_time = end_time - start_time
        #print(f"\n-> Jacobi Tempo: {self.execution_time:.4f} secondi")
        #print(f"-> Errore finale: {np.linalg.norm(w - w_old):.6f}")
        #print(f"-> Iterazioni totali: {iter_count}")

        # Rilascio della memoria condivisa
        shm.close()
        shm.unlink()
        
        return loss_history, gradient_history,gradient_norm_history, w, iter_count

    def compute_loss_J(self, X, y, w):
        return compute_loss(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)
    
    def compute_grad_J(self, X, y, w):  # Rinominato da compute_gradient_J a compute_grad_J
        return compute_grad(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)
    