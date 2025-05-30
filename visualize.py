import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import os

def plot_losses_iter_overlay(loss_histories, method_names, save_path):
    """
    Crea un grafico unico con le loss sovrapposte per più metodi, ognuno con un colore specifico.

    Parameters:
    - loss_histories: lista di array o liste contenenti le loss di ciascun metodo
    - method_names: lista dei nomi dei metodi corrispondenti
    - save_path: percorso dove salvare l'immagine risultante
    """

    plt.figure(figsize=(10, 6))

    # Mappa colore per metodo (modificabile a piacere)
    color_map = {
        "Gauss-Seidel": "blue",
        "Jacobi": "green",
        "Gradient Descent": "red",
        "Gradient Descent Armijo": "purple"
    }

    for loss, name in zip(loss_histories, method_names):
        color = color_map.get(name, None)  # usa colore se definito, altrimenti default
        plt.plot(loss, label=name, color=color)

    plt.xlabel("Epoche / Iterazioni")
    plt.ylabel("Loss")
    plt.title("Confronto delle Loss tra i Metodi")
    plt.legend(title="Metodo", loc="best")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_losses_time_overlay(loss_histories, time_histories, method_names, save_path):
    """
    Crea un grafico unico con le loss sovrapposte in funzione del tempo (Loss vs Tempo)
    per più metodi, ognuno con un colore specifico.

    Parameters:
    - loss_histories: lista di liste o array, ogni elemento è la loss per un metodo
    - time_histories: lista di liste o array, ogni elemento è la lista dei tempi cumulati per un metodo
    - method_names: lista dei nomi dei metodi corrispondenti
    - save_path: percorso dove salvare l'immagine risultante
    """
    plt.figure(figsize=(10, 6))

    # Mappa colore per metodo (può essere personalizzata)
    color_map = {
        "Gauss-Seidel": "blue",
        "Jacobi": "green",
        "Gradient Descent": "red",
        "Gradient Descent Armijo": "purple"
    }

    for loss, time, name in zip(loss_histories, time_histories, method_names):
        color = color_map.get(name, None)
        plt.plot(time, loss, label=name, color=color)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Loss")
    plt.title("Confronto delle Loss nel Tempo tra i Metodi")
    plt.legend(title="Metodo", loc="best")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# Funzione per visualizzare e salvare le loss e i gradienti
def plot_loss_and_gradient(loss_history,loss_history_block, gradient_history, gradient_norm_slope,gradient_slope, method_name, save_dir):
    # Creare la cartella se non esiste
    os.makedirs(save_dir, exist_ok=True)
    #Creo una figura con 4 subplot, 2 righe e 2 colonne
    plt.figure(figsize=(12, 10))

    # Determina l'etichetta dell'asse delle ascisse in base al metodo
    x_label = "Iterazioni sulla funzione totale" if method_name in ["Gauss-Seidel", "Jacobi"] else "Epoca"

    #Prima Riga

    # Grafico delle loss
    plt.subplot(2, 2, 1)
    plt.plot(loss_history, label="Loss", color="blue")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.title(f"Loss - {method_name}")
    plt.legend()
    plt.grid(True)

    # Grafico della norma del gradiente
    plt.subplot(2, 2, 2)
    plt.plot(gradient_history, label="Norma del Gradiente", color="red")
    plt.xlabel(x_label)
    plt.ylabel("Norma del Gradiente")
    plt.title(f"Gradiente - {method_name}")
    plt.legend()
    plt.grid(True)

    #Seconda riga
    plt.subplot(2,2,3)
    if method_name=="Gauss-Seidel":
        #plot del grafico della loss con valori ritrovati dentro i blocchi 
        plt.plot(loss_history_block, label="Loss dentro ai blocchi", color="green")
        plt.xlabel("Iterazioni su ogni sottoblocco della funzione")
        plt.ylabel("Loss")
        plt.title(f"Loss dentro a tutti i blocchi- {method_name}")
        plt.legend()
        plt.grid(True)

    else:
        # Plot della Pendenza della Norma del Gradiente
        plt.plot(gradient_norm_slope, color="green")
        plt.xlabel(x_label)
        plt.ylabel("Pendenza della Norma del Gradiente")
        plt.title(f"Pendenza della norma del Gradiente\n- {method_name}")
        plt.grid(True)
        
        # Aggiungi descrizione come annotazione
        plt.annotate('Misura la velocità di cambiamento\n della norma del gradiente',
                    xy=(0.5, -0.25), xycoords='axes fraction',
                    ha='center', fontsize=10, bbox=dict(boxstyle="round", fc="w"))

    # Plot della Pendenza dei vettori del Gradiente
    plt.subplot(2, 2, 4)
    gradient_slope = np.array(gradient_slope)
    
    if gradient_slope.ndim == 2:
        sns.heatmap(gradient_slope.T, cmap="coolwarm", cbar_kws={'label': 'Δ Gradiente'})
        plt.title(f"Heatmap della Pendenza dei Vettori del Gradiente\n- {method_name}")
        plt.xlabel("Epoche")
        plt.ylabel("Singole componenti del Gradiente")
    else:
        # fallback: se è 1D, fai comunque un plot standard
        plt.plot(gradient_slope, label="Δ Gradiente", color="purple")
        plt.title(f"Pendenza dei Vettori del Gradiente\n- {method_name}")
        plt.xlabel("Epoche")
        plt.ylabel("Δ Gradiente")
        plt.legend()
    
            
        '''
            In Gauss seidel np.diff() calcola la differenza tra ogni coppia di iterazioni successive, 
            quindi restituisce un array con una dimensione in meno. Questo perchè gradient_history ha solo tre elementi 
            e quindi ne restituisce due
        '''

    # Aggiungi descrizione come annotazione
    plt.annotate(f'Mostra come ogni componente del\n gradiente cambia nelle epoche',
                xy=(0.5, -0.25), xycoords='axes fraction',
                ha='center', fontsize=10, bbox=dict(boxstyle="round", fc="w"))
  
    
    # Regola il margine inferiore per accomodare l'annotazione
    plt.subplots_adjust(bottom=0.15)
    # Titolo generale
    plt.suptitle(f"Analisi di Training - Metodo {method_name}", y=1.02, fontsize=14)
    plt.tight_layout()

    # Costruzione del nome del file
    base_filename = f"{method_name}_plot"
    extension = ".png"
    save_path = os.path.join(save_dir, base_filename+"1"+ extension)

    # Se il file esiste, trova un nome unico aggiungendo un numero progressivo
    counter = 2
    while os.path.exists(save_path):
        save_path = os.path.join(save_dir, f"{base_filename}_{counter}{extension}")
        counter += 1

    # Salva il grafico con il nome unico
    plt.savefig(save_path)
    
    plt.close()  # Chiude la figura per risparmiare memoria
    return save_path