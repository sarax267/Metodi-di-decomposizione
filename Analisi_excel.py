from tqdm import tqdm
from analysis_prova import Analisi
from save_results import save_results_to_excel
import pandas as pd
file_excel = "risultati.xlsx"
foglio1="Risultati"
foglio2="Analisi"
# Creazione di una barra di avanzamento personalizzata per l'analisi
with tqdm(total=1, desc="Analisi", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
    df=Analisi(file_excel,foglio1)
    pbar.update(1)  # Completa la barra
    print("\nFile excel analizzato")  # \n per nuova linea dopo la barra
    #save_results_to_excel(df, file_excel,foglio2)