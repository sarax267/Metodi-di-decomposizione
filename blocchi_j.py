import numpy as np
from loss import compute_loss
from armijo_line_search import ArmijoLineSearch
from gradient import compute_grad



def blocchi_j(X, y, w, start_idx, end_idx, lambda_reg, max_iter, tol,max_iter_in_block):
    
    #print(f"Dimensioni X: {X.shape[0]} righe e {X.shape[1]} colonne")
    #print(f"Dimensioni y: {y.shape[0]} righe")
    #print(f"Dimensioni w: {w.shape[0]} righe")
    #print(f"Partenza: {start_idx} Arrivo:{end_idx}")
    #Estraggo il blocco corrente
    #print(f"w:{w}")
    X_block = X[:, start_idx:(end_idx+1)]
    w_block = w[start_idx:(end_idx+1)].copy()
    #print(f"w_block:{w_block}")
    #print(f"Dimensioni X_block: {X_block.shape[0]} righe e {X_block.shape[1]} colonne")
    #Estraggo il blocco precedente
    X_before = X[:, :start_idx]
    w_before = w[:start_idx]
    #print(f"w_before:{w_before}")
    #print(f"Dimensioni X_before: {X_before.shape[0]} righe e {X_before.shape[1]} colonne")
    #Estraggo il blocco successivo
    X_after = X[:, (end_idx+1):]
    w_after = w[(end_idx+1):]
    #print(f"w_after: {w_after}")
    #print(f"Dimensioni X_after: {X_after.shape[0]} righe e {X_after.shape[1]} colonne")
    
    # Calcolo dei contributi iniziali
    z_block = X_block @ w_block
    z_before = X_before @ w_before if start_idx > 0 else 0
    z_after = X_after @ w_after if end_idx < X.shape[1] else 0

    method_Armijo = ArmijoLineSearch()
    
    for iter_num in range(max_iter_in_block):
        
        # Definisco la funzione di loss per questo blocco specifico
        def block_loss(w_new):
            z_new = X_block @ w_new
            return compute_loss(X_block, y, w_new, z_new, z_before, z_after, lambda_reg)
        
        
        # Aggiorna w_block in base al metodo scelto (Jacobi o Gauss-Seidel)
        
        # Calcola il gradiente per il blocco corrente
        grad = compute_grad(X_block, y, w_block, z_block, z_before, z_after, lambda_reg)
        #print(f"Dimensione gradiente: {grad.shape[0]}")
        grad_norm = np.linalg.norm(grad) 
        #Verifico la convergenza
        if grad_norm  < tol:
            break
        direction = -grad  # Direzione di discesa
        # Armijo con ricalcolo di z_block
        alpha = method_Armijo.search(
            lambda w_new: block_loss(w_new), 
            grad, X_block, y, w_block, direction
        )
        # Aggiorno tutti i pesi nel blocco simultaneamente
        w_block = w_block + alpha * direction
        z_block = X_block @ w_block  # Ricalcolo z_block con il nuovo w_block
        
    w_tot=w.copy()
    # Aggiorna il vettore w con i nuovi pesi del blocco
    
    #print(f"w aggiornatao: {w_tot}")
    #print(f"Dimensioni w_tot: {w_tot.shape[0]}")
    # Calcola il gradiente e la loss totali con z_total aggiornato
   

    return w_block,iter_num
