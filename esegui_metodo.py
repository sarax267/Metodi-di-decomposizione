
from gradient_descent import GradientDescent
from gradient_descent_armijo import GradientDescentArmijo
from gauss_seidel import GaussSeidel
from jacobi import Jacobi

from create_dataframe import create_results_dataframe
from  visualize import plot_loss_and_gradient 
import numpy as np

def Metodi(name_dataset, method,joblib_num_jobs, X,y, Caratteristiche, Campioni, file_excel):
    #Dati
    lr=0.1 #Parro di Armijo fisso
    epochs=100
    #Dati
    num_blocks=2
    lambda_reg=0.1
    tol=1e-6
    max_iter=100
    num_jobs=joblib_num_jobs
    max_iter_in_block=10
    
    if method =="gradient_descent":
        
        model= GradientDescent(lambda_reg, lr, epochs,tol)

        Loss_complete,Gradient_norm_complete, Gradient_complete,w,iter,Time=model.fit(X, y)
       
        #pendenza della norma del gradiente
        gradient_norm_slope_gd = np.diff(Gradient_norm_complete)
        #pendenza del vettore dei gradienti 
        gradient_slope_gd = [Gradient_complete[i] - Gradient_complete[i - 1] 
                for i in range(1, len(Gradient_complete))]

        path_img=plot_loss_and_gradient(Loss_complete,0, Gradient_norm_complete,gradient_norm_slope_gd,gradient_slope_gd, "Gradient Descent",save_dir="plots_"+name_dataset)

        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Gradient Descent", iter, model.execution_time, Loss_complete[-1], Gradient_norm_complete[-1],path_img, num_sub_block=1)
        

    if method =="gradient_descent_armijo":
        
        model = GradientDescentArmijo(lambda_reg, epochs,tol)

        Loss_complete,Gradient_norm_complete,Gradient_complete, w,iter,Time=model.fit(X, y)
       
        #pendenza della norma del gradiente
        gradient_norm_slope_gda = np.diff(Gradient_norm_complete)
        #pendenza del vettore dei gradienti 
        gradient_slope_gda = [Gradient_complete[i] - Gradient_complete[i - 1] 
                for i in range(1, len(Gradient_complete))]

        path_img=plot_loss_and_gradient(Loss_complete,0, Gradient_norm_complete,gradient_norm_slope_gda,gradient_slope_gda, "Gradient Descent Armijo",save_dir="plots_"+name_dataset)
        
        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Gradient Descent Armijo", iter, model.execution_time, Loss_complete[-1], Gradient_norm_complete[-1], path_img,num_sub_block=1)
        

    if method =="jacobi":
        #print("---------------------------------------------------------------")
        #print("Metodo di Jacobi")

        model = Jacobi(num_blocks, lambda_reg, tol, max_iter,num_jobs, max_iter_in_block)
        
        Loss_complete,Gradient_complite,Gradient_norm_complete,w_opt,iterazioni,Time = model.jacobi(X, y)
        
        #pendenza della norma del gradiente
        gradient_norm_slope_j = np.diff(Gradient_norm_complete)

        #pendenza del vettore dei gradienti 
        gradient_slope_j = [Gradient_complite[i] - Gradient_complite[i - 1]  # Forza array 1D
            for i in range(1, len(Gradient_complite))
        ]
      
        path_img=plot_loss_and_gradient(Loss_complete,0, Gradient_norm_complete,gradient_norm_slope_j,gradient_slope_j, "Jacobi",save_dir="plots_"+name_dataset)
        
        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Jacobi", iterazioni, model.execution_time, Loss_complete[-1], Gradient_norm_complete[-1],path_img,num_blocks)
       
    if method =="gauss_seidel":
    
        model = GaussSeidel(num_blocks, lambda_reg, tol, max_iter, max_iter_in_block)
        
        #print("** Applico il metodo **")
        Loss_complete,Loss_block,Gradient_norm_complete,Gradient_complete,iterazioni,Time = model.gauss_seidel(X, y)
    
        #pendenza della norma del gradiente
        gradient_norm_slope_gs = np.diff(Gradient_norm_complete)
        #pendenza del vettore dei gradienti 
        for i in range(1, len(Gradient_complete)):
                gradient_slope_gs =[Gradient_complete[i] - Gradient_complete[i - 1]]
       
        path_img=plot_loss_and_gradient(Loss_complete,Loss_block, Gradient_norm_complete,gradient_norm_slope_gs,gradient_slope_gs, "Gauss-Seidel",save_dir="plots_"+name_dataset)
      
        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Gauss-Seidel", iterazioni, model.execution_time, Loss_complete[-1], Gradient_norm_complete[-1],path_img,num_blocks)
       
    return results_df,Loss_complete,Time