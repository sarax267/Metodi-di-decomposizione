import numpy as np
from tqdm import tqdm
from analysis import Analisi
import argparse
import time
import pandas as pd
import os
from save_results import salva_matrice
from sklearn.linear_model import LogisticRegression
from adatta_dati import estrai_dati
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine, fetch_20newsgroups, fetch_openml
from save_results import save_results_to_excel
from esegui_metodo import Metodi
from visualize import plot_losses_iter_overlay, plot_losses_time_overlay

if __name__ == "__main__":
    # Apri (o crea) un file in modalità scrittura
    with open("output.txt", "a") as f:
        # Configura l'argomento da riga di comando
        parser = argparse.ArgumentParser(description="Esegui metodi di ottimizzazione.")
        parser.add_argument("--method", nargs="+", choices=["gradient_descent", "gradient_descent_armijo", "gauss_seidel", "jacobi"], help="Metodi da eseguire.")
        args = parser.parse_args()

        '''Dataset 1: Breast_cancer'''
        name_dataset_1="Breast_Cancer"
        # Caricamento e preprocessing
        breast_cancer = load_breast_cancer()
        X_breast_cancer,y_breast_cancer,Campioni_breast_cancer,Caratteristiche_breast_cancer,data_breast_cancer= estrai_dati(name_dataset_1,breast_cancer)
        
        '''Dataset 2: Diabetes'''
        name_dataset_2="Diabetes"
        # Caricamento e preprocessing
        diabetes = load_diabetes()
        X_diabetes,y_diabetes,Campioni_diabetes,Caratteristiche_diabetes,data_diabetes= estrai_dati(name_dataset_2,diabetes)

        '''Dataset 3: Digits'''
        name_dataset_3="Digits"
        # Caricamento e preprocessing
        digits = load_digits()
        X_digits,y_digits,Campioni_digits,Caratteristiche_digits,data_digits= estrai_dati(name_dataset_3,digits)
        
        '''Dataset 4: Iris'''
        name_dataset_4="Iris"
        # Caricamento e preprocessing
        iris = load_iris()
        X_iris,y_iris,Campioni_iris,Caratteristiche_iris,data_iris= estrai_dati(name_dataset_4,iris)
    

        '''Dataset 5: Wine'''
        name_dataset_5="Wine"
        # Caricamento e preprocessing
        wine = load_wine()
        X_wine,y_wine,Campioni_wine,Caratteristiche_wine,data_wine= estrai_dati(name_dataset_5,wine)
    
        '''Dataset 6: Fetch'''
        name_dataset_6="Fetch"
        # Caricamento e preprocessing
        categories = ['alt.atheism', 'comp.graphics']  # Scegli due categorie qualsiasi
        fetch = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
        X_fetch,y_fetch,Campioni_fetch,Caratteristiche_fetch,data_fetch= estrai_dati(name_dataset_6,fetch)

        '''Dataset 7: Adult'''
        name_dataset_7="Adult" # persone in america che guadagnano > o < 50k
        # Caricamento e preprocessing
        #adult = fetch_openml(name='adult', version=2, as_frame=True)
        column_names = [  # inserisci qui i nomi corretti se vuoi
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]

        adult = pd.read_csv('adult/adult.data', names=column_names, na_values=' ?', skipinitialspace=True)
        X_adult,y_adult,Campioni_adult,Caratteristiche_adult,data_adult= estrai_dati(name_dataset_7,adult)

        
        #Creo il file per il salvataggio dei dati    
        file_excel = "risultati.xlsx"
        foglio1="Risultati"
        foglio2="Analisi"
        messaggi=[
            f"Il dataset usato è Breast cancer di Sklearn e considera:\n{Campioni_breast_cancer} campoioni\n{Caratteristiche_breast_cancer} features\n ogni campione ha una label =-1 se maligno =1 se benigno\n",
            f"Il dataset usato è Diabetes di Sklearn e considera:\n{Campioni_diabetes} campoioni\n{Caratteristiche_diabetes} features\n ogni campione ha una label =-1 se inferiore a una certa soglia =1 se superiore\n",
            f"Il dataset usato è Digits di Sklearn e considera:\n{Campioni_digits} campoioni\n{Caratteristiche_digits} features\n ogni campione ha una label =-1 se pari =1 se dispari\n",
            f"Il dataset usato è Iris di Sklearn e considera:\n{Campioni_iris} campoioni\n{Caratteristiche_iris} features\n ogni campione ha una label =-1 se appartenete alle categorie A e B =1 se appartenete alla categoria C\n",
            f"Il dataset usato è Wine di Sklearn e considera:\n{Campioni_wine} campoioni\n{Caratteristiche_wine} features\n ogni campione ha una label =-1 se appartenete alle categorie A e B =1 se appartenete alla categoria C\n\n",
            f"Il dataset usato è fetch di Sklearn e considera:\n{Campioni_fetch} campoioni\n{Caratteristiche_fetch} features\n ogni campione ha una label =-1 se inferiore a una certa soglia =1 se superiore\n",
            f"Il dataset usato è adult di Sklearn e considera:\n{Campioni_adult} campoioni\n{Caratteristiche_adult} features\n ogni campione ha una label =-1 se inferiore a 50k =1 se superiore a 50k\n"
            ]
        
        # Lista dei dati e nomi dei dataset
        dati_e_nomi = [
            (data_breast_cancer, name_dataset_1),
            (data_diabetes, name_dataset_2),
            (data_digits, name_dataset_3),
            (data_iris, name_dataset_4),
            (data_wine, name_dataset_5),
            (data_fetch, name_dataset_6),
            (data_adult, name_dataset_7)
        ]

        # Salvataggio con barra di avanzamento
        with tqdm(dati_e_nomi, desc="Salvataggio matrici", unit="dataset") as pbar:
            for dati, nome_dataset in pbar:
                salva_matrice(dati, nome_dataset, file_excel)
                pbar.set_postfix({"dataset": nome_dataset})

        print("✅ Tutti i dataset sono stati salvati")
        print("\n")

        # Crea un dataframe vuoto
        df_finale = pd.DataFrame()

        # Lista di tuple con (nome_dataset, variabili) per ogni dataset
        datasets = [
            (name_dataset_1, X_breast_cancer, y_breast_cancer, Caratteristiche_breast_cancer, Campioni_breast_cancer),
            (name_dataset_2, X_diabetes, y_diabetes, Caratteristiche_diabetes, Campioni_diabetes),
            (name_dataset_3, X_digits, y_digits, Caratteristiche_digits, Campioni_digits),
            (name_dataset_4, X_iris, y_iris, Caratteristiche_iris, Campioni_iris),
            (name_dataset_5, X_wine, y_wine, Caratteristiche_wine, Campioni_wine),
            (name_dataset_6, X_fetch, y_fetch, Caratteristiche_fetch, Campioni_fetch),
            (name_dataset_7, X_adult, y_adult, Caratteristiche_adult, Campioni_adult)
        ]
        
        losses=[]
        Time_losses=[]
        # Itera su messaggi e dataset insieme
        for messaggio, (name, X, y, caratteristiche, campioni) in tqdm(zip(messaggi, datasets), 
                                                                    total=len(messaggi),
                                                                    desc="Elaborazione dataset"):
            print(messaggio, file=f)
            
            for method in tqdm(args.method, desc="Metodi", leave=False):
                if method == "jacobi":
                    os.environ["JOBLIB_NUM_JOBS"] = "4"
                else:
                    os.environ["JOBLIB_NUM_JOBS"] = "1"

                joblib_num_jobs = int(os.getenv("JOBLIB_NUM_JOBS", 1))
                #print(f"\nEsecuzione metodo {method} con {joblib_num_jobs} CPU:", file=f)
            
                '''Eseguo i metodi SOLO per il dataset corrente'''
                risultato,loss,Time = Metodi(name, method, joblib_num_jobs, X, y, caratteristiche, campioni, file_excel)
                losses.append(loss)
                Time_losses.append(Time)
                # Aggiungi solo il risultato corrente al dataframe finale
                df_finale = pd.concat([df_finale, risultato], ignore_index=True)

            path_loss_iter_overlay=plot_losses_iter_overlay(losses, args.method, save_path=f"plots_{name}/loss__iter_overlay.png")
            path_loss_time_overlay=plot_losses_time_overlay(losses, Time_losses,args.method, save_path=f"plots_{name}/loss__time_overlay.png")
            # Aggiungo una riga: Loss vs iterazioni
            row_overlay_iter = {col: "" for col in df_finale.columns}  # inizializza tutto vuoto
            row_overlay_iter["Plot"] = path_loss_iter_overlay
            row_overlay_iter["Loss Finale"] = "Loss vs iter overlay"
            df_finale = pd.concat([df_finale, pd.DataFrame([row_overlay_iter])], ignore_index=True)
            # Seconda riga: Loss vs Tempo overlay
            row_overlay_time = {col: "" for col in df_finale.columns}
            row_overlay_time["Plot"] = path_loss_time_overlay
            row_overlay_time["Loss Finale"] = "Loss vs time overlay"
            df_finale = pd.concat([df_finale, pd.DataFrame([row_overlay_time])], ignore_index=True)

        #print(df_finale)
        # Creazione di una barra di avanzamento personalizzata per il salvataggio
        with tqdm(total=1, desc="Salvataggio risultati", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
            save_results_to_excel(df_finale,file_excel,foglio1)
            pbar.update(1)  # Completa la barra
            print("\nFile salvato")  # \n per nuova linea dopo la barra

        
        '''
        merged_df = pd.merge(results_df_gd,results_df_gda,results_df_gs,results_df_j)
        analisi=Analisi(merged_df)
        save_analysis_to_excel(analisi)
        '''
        