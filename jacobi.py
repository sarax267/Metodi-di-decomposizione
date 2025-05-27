import numpy as np
from gradient import compute_grad
from loss import compute_loss
from blocchi_j import blocchi_j
import time
from joblib import Parallel, delayed
from multiprocessing.shared_memory import SharedMemory

class Jacobi:
    def __init__(self, num_blocks, lambda_reg, tol, max_iter, num_jobs, max_iter_in_block):
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.max_iter = max_iter
        self.num_blocks = num_blocks
        self.num_jobs = num_jobs
        self.max_iter_in_block = max_iter_in_block
        self.execution_time = 0

    def jacobi(self, X, y):
        with open("output.txt", "a") as f:
            np.set_printoptions(precision=3)
            n_features = X.shape[1]

            # memoria condivisa per w
            shm = SharedMemory(create=True, size=n_features * np.float64().itemsize)
            w = np.ndarray(n_features, dtype=np.float64, buffer=shm.buf)
            w.fill(0)

            loss_finale = []
            gradient_finale = []
            gradient_norm_finale = []

            z = X @ w
            grad = compute_grad(X, y, w, z, 0, 0, self.lambda_reg)
            loss = compute_loss(X, y, w, z, 0, 0, self.lambda_reg)

            gradient_finale.append(grad)
            gradient_norm_finale.append(np.linalg.norm(grad))
            loss_finale.append(loss)

            block_size = n_features // self.num_blocks
            start_time = time.perf_counter()

            for iteration in range(self.max_iter):
                # Prepara il calcolo parallelo
                def update_block(i):
                    start_idx = i * block_size
                    if i == self.num_blocks - 1:
                        end_idx = start_idx + block_size - 1 + (n_features % self.num_blocks)
                    else:
                        end_idx = start_idx + block_size - 1

                    try:
                        updated_w_block, iters = blocchi_j(
                            X, y, w.copy(), start_idx, end_idx,
                            self.lambda_reg, self.max_iter,
                            self.tol, self.max_iter_in_block
                        )
                        # Calcola la loss corrispondente al blocco aggiornato
                        w_temp = w.copy()
                        w_temp[start_idx:end_idx + 1] = updated_w_block
                        z_temp = X @ w_temp
                        loss_temp = compute_loss(X, y, w_temp, z_temp, 0, 0, self.lambda_reg)
                        return loss_temp, updated_w_block, start_idx, end_idx
                    except Exception as e:
                        print(f"Errore nel blocco {i}: {e}")
                        return None

                results = Parallel(n_jobs=self.num_jobs)(
                    delayed(update_block)(i) for i in range(self.num_blocks)
                )

                # Filtra risultati validi
                results = [res for res in results if res is not None]
                if not results:
                    break  # Nessun aggiornamento possibile

                # Scegli il blocco con loss minima
                best_idx = min(range(len(results)), key=lambda i: results[i][0])
                _, best_w_block, best_start_idx, best_end_idx = results[best_idx]

                # Aggiorna solo il blocco selezionato
                w[best_start_idx:best_end_idx + 1] = best_w_block

                # Calcola loss, gradiente e norma globale
                z = X @ w
                grad = compute_grad(X, y, w, z, 0, 0, self.lambda_reg)
                grad_norm = np.linalg.norm(grad)
                loss = compute_loss(X, y, w, z, 0, 0, self.lambda_reg)

                loss_finale.append(loss)
                gradient_finale.append(grad)
                gradient_norm_finale.append(grad_norm)

                if grad_norm < self.tol:
                    break

            print(gradient_norm_finale, file=f)

            self.execution_time = time.perf_counter() - start_time
            shm.close()
            shm.unlink()

        return loss_finale, gradient_finale, gradient_norm_finale, w, iteration

    def compute_loss_J(self, X, y, w):
        return compute_loss(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)

    def compute_grad_J(self, X, y, w):
        return compute_grad(X, y, w, z_block=0, z_before=0, z_after=0, lambda_reg=self.lambda_reg)
