import numpy as np
from blocchi_gs import blocchi_gs
import time
from loss import compute_loss



class GaussSeidel:
    def __init__(self, num_blocks,lambda_reg, tol, max_iter):
        self.lambda_reg = lambda_reg  # Coefficiente di regolarizzazione
        self.tol = tol  # Tolleranza per la convergenza
        self.max_iter = max_iter  # Numero massimo di iterazioni
        self.num_blocks= num_blocks #Numero di Blocchi
        self.execution_time=0
        self.epochs=200


    def gauss_seidel(self, X, y):
        """Metodo di Gauss-Seidel"""

        # Impostazione globale per stampare i vettori con 3 decimali
        np.set_printoptions(precision=3)

        # Inizializzazione dei pesi
        w = np.zeros(X.shape[1])
        loss_history = [] #loss globale, dopo ogni ottimizzazione di blocco
        gradient_history = []
        gradient_norm_history=[]
        w_history=[]
        iter_history=[]
        iter_count=0 #Voglio contare il numero di iterazioni che il metodo impiega prima di fermarsi per il criterio di arresto

        n_features = X.shape[1] # mumero delle colonne di X= numero dello delle features
        block_size = n_features // self.num_blocks #calcolo la dimensione di ogni blocco, e prendo l'intero della divisione
        #print(f"Numero delle features {n_features}")
        #print(f"Numero dei blocchi:{self.num_blocks}")
        #print(f"Dimensione dei blocchi:{block_size}")

        start_time = time.perf_counter()  # Misura il tempo

        w_current = w.copy()  # Vettore dei pesi corrente
        
        # Calcola la loss iniziale prima di iniziare l'ottimizzazione
        z_initial = X @ w_current
        initial_loss = compute_loss(X, y, w_current, z_initial, 0, 0, self.lambda_reg)
        loss_history.append(initial_loss)

        for epoch in range(self.max_iter): #Ciclo esterno
            max_grad_norm=0
            epoch_converged = True  # Flag per verificare la convergenza in questo epoch

            for i in range(self.num_blocks):
                #print(f"Sono nel blocco numero {i+1}")
                start_idx = i * (block_size) #Qui mi dice dove parte un blocco. 
                    #print(f"Indice di partenza del blocco {i} è :{start_idx}")
                        #Es i=2, quindi sono nel secondo blocco e la lunghezza di ogni blocco è 3, ho che il secondo blocco parte dalla posizione 3
                        #perchè le posizioni del vettore saranno 0 1 2 | 3 4 5 | 6 7 8 |9 10 11|...
                if i==self.num_blocks-1: # L'ultimo blocco raccoglie i termini in più se ci sono
                    end_idx = start_idx+block_size-1+(n_features % self.num_blocks) #Qui dove finisce un blocco 
                        
                else: 
                    end_idx = start_idx+block_size-1 #Qui dove finisce un blocco 
                        
                    #print(f"Start_index: {start_idx}")
                    #print(f"End_index: {end_idx}")
                    # Calcola la discesa del gradiente solo su questo blocco
                
                #Ottimizzo il blocco
                loss,grad,grad_norm,w_return,iter_count,loss_block,grad_norm_block=blocchi_gs(X,y,w_current,start_idx,end_idx,self.lambda_reg,self.max_iter,self.tol)
                w_current=w_return.copy()
                #aggiorno il contatore
                iter_count+=len(loss_block)
                #memorizzo le metriche per ogni iterazione all'interno del blocco
                if len(loss_block) > 0:  # Evita di aggiungere liste vuote
                    loss_history.extend(loss_block)
                    gradient_norm_history.extend(grad_norm_block)
                
                max_grad_norm=max(max_grad_norm,grad_norm)
                
                #Se un blocco non converge, l'itera epoca non converge
                if grad_norm>=self.tol:
                    epoch_converged=False
                # Controllo della convergenza/Criterio di arresto
                #print(f"Tolleranza impostata per Gauss Seidel: {self.tol}")
                #print(f"Errore: {np.linalg.norm(w - w_old)} all'iterazione {iter_count}")
                #e fermare l'iterazione se i pesi w non stanno cambiando significativamente tra un'iterazione e l'altra

                #Verifica della compattezza dell'insieme di livello:
                # Controllo della norma di w (limitazione)
                current_norm = np.linalg.norm(w_current)
                if current_norm > 1e6:
                    raise ValueError("Errore: ||w|| sta divergendo. L'insieme di livello potrebbe non essere limitato.")

            # Controllo della loss (chiusura)
            if np.isnan(loss) or np.isinf(loss):
                raise ValueError("Errore: La loss è divergente. L'insieme di livello potrebbe non essere chiuso.")
            
            #Memorizzo dopo ogni epoca
            gradient_history.append(grad.copy())
            w_history.append(w_current.copy())
            iter_history.append(iter_count)
            if epoch_converged and max_grad_norm< self.tol:
                break #Convergenza
        #print(f"Numero di iterazioni{iter_history}")
    
        
        #print(f"Pesi blocchi: {w_history}")
        end_time = time.perf_counter() # Fine del timer
        self.execution_time=end_time - start_time
        #print(f"\n -> Tempo di esecuzione: {self.execution_time} secondi")
        #print(f"\n -> Converge con un errore finale di {np.linalg.norm(w - w_old):.3f} dopo {iter_count} iterazioni")
        return loss_history,gradient_norm_history, gradient_history, w_current, iter_history  # Restituisce le loss, i gradienti e i pesi finali