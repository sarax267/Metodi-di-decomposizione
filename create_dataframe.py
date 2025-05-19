import pandas as pd

def create_results_dataframe(Cpu, name_dataset, Caratteristiche, Campioni, metodo, epoche, tempo, loss_finale, gradiente_finale, plot_path, num_sub_block):
    """
    Crea un dataframe pandas con i risultati dell'esecuzione.
    
    Args:
        Cpu: Numero di CPU utilizzate
        Name_dataset: Nome del dataset dei dati
        Caratteristiche: Numero di caratteristiche
        Campioni: Numero di campioni
        metodo: Metodo utilizzato
        epoche: Numero di epoche/iterazioni
        tempo: Tempo di esecuzione in secondi
        loss_finale: Valore finale della loss
        gradiente_finale: Valore finale del gradiente
        plot_path: Percorso del file del plot
        num_sub_block= Numero dei sottoblocchi usati in Gs e J
        
    Returns:
        pd.DataFrame: Dataframe con i risultati
    """
    colonne_complete = [
        "Esecuzione","Dataset", "Metodo", "Epoche/Iterazioni",
        "Tempo di Esecuzione (s)", "Loss Finale", 
        "Gradiente Finale", "Plot", "Commenti", 
        "Numero caratteristiche", "Numero Campioni X", "CPU usate", "Numero sottoblocchi"
    ]

    # Crea il dataframe con i nuovi dati
    nuovi_dati = pd.DataFrame(columns=colonne_complete)
    nuovi_dati.loc[0] = [
        None,  # Esecuzione verrà aggiornata dopo
        name_dataset,
        metodo,
        epoche,
        tempo,
        loss_finale,
        gradiente_finale,
        plot_path if plot_path else "N/A",
        None,  # Commenti da aggiungere manualmente
        Caratteristiche,
        Campioni,
        Cpu,
        num_sub_block,
    ]
    
    return nuovi_dati