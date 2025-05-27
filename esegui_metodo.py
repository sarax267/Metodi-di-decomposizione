
from gradient_descent import GradientDescent
from gradient_descent_armijo import GradientDescentArmijo
from gauss_seidel import GaussSeidel
from jacobi import Jacobi

from create_dataframe import create_results_dataframe
from  visualize import plot_loss_and_gradient 
import numpy as np

def Metodi(name_dataset, method,joblib_num_jobs, X,y, Caratteristiche, Campioni, file_excel):
    #Dati
    lr=0.1
    epochs=200
    #Dati
    num_blocks=8
    lambda_reg=0.1
    tol=1e-6
    max_iter=100
    num_jobs=joblib_num_jobs
    max_iter_in_block=10
    
    if method =="gradient_descent":
        #print("---------------------------------------------------------------")   
        
        model_grad_desc = GradientDescent(lambda_reg, lr, epochs)
        Loss_gradient_descent_complete,Gradient_norm_gradient_descent_complete, Gradient_gradient_descent_complete,w=model_grad_desc.fit(X, y)
        #pendenza della norma del gradiente
        gradient_norm_slope_gd = np.diff(Gradient_norm_gradient_descent_complete)
        #pendenza del vettore dei gradienti 
        gradient_slope_gd = [Gradient_gradient_descent_complete[i] - Gradient_gradient_descent_complete[i - 1] 
                for i in range(1, len(Gradient_gradient_descent_complete))]

        #print("\n  -> Loss finale con il metodo del gradient descent:", Loss_gradient_descent_complete[-1])
        #print("\n  -> Gradiente finale con il metodo del gradient descent:\n", Gradient_gradient_descent_complete[-1])
        #print("\n  -> Vettore dei vettori dei gradienti con il metodo del gradient descent:\n", Gradient_gradient_descent_complete)
        #print("\n  -> Vettore delle norme dei gradienti con il metodo del gradient descent:\n", Gradient_norm_gradient_descent_complete)
        path_img_GD=plot_loss_and_gradient(Loss_gradient_descent_complete,0, Gradient_norm_gradient_descent_complete,gradient_norm_slope_gd,gradient_slope_gd, "Gradient Descent",save_dir="plots_"+name_dataset)

        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Gradient Descent", epochs, model_grad_desc.execution_time, Loss_gradient_descent_complete[-1], Gradient_norm_gradient_descent_complete[-1],path_img_GD, num_sub_block=1)
        #save_results_to_excel(name_dataset,results_df_gd, file_excel)

    if method =="gradient_descent_armijo":
        #print("---------------------------------------------------------------")


        model_grad_desc_Arm = GradientDescentArmijo(lambda_reg, epochs)
        Loss_gradient_descent_armijo_complete,Gradient_norm_gradient_descent_armijo_complete,Gradient_gradient_descent_armijo_complete, w=model_grad_desc_Arm.fit(X, y)

        #pendenza della norma del gradiente
        gradient_norm_slope_gda = np.diff(Gradient_norm_gradient_descent_armijo_complete)
        #pendenza del vettore dei gradienti 
        gradient_slope_gda = [Gradient_gradient_descent_armijo_complete[i] - Gradient_gradient_descent_armijo_complete[i - 1] 
                for i in range(1, len(Gradient_gradient_descent_armijo_complete))]

        #print("\n  -> Loss finale con il metodo del gradient descent armijo:", Loss_gradient_descent_armijo_complete[-1])
        #print("\n  -> Gradiente finale con il metodo del gradient descent armijo:\n", Gradient_gradient_descent_armijo_complete[-1])
        #print("\n  -> Vettore dei vettori dei gradienti con il metodo del gradient descent armijo:\n", Gradient_gradient_descent_armijo_complete)
        #print("\n  -> Vettore delle norme dei gradienti con il metodo del gradient descent armijo:\n", Gradient_norm_gradient_descent_armijo_complete)
        path_img_GDA=plot_loss_and_gradient(Loss_gradient_descent_armijo_complete,0, Gradient_norm_gradient_descent_armijo_complete,gradient_norm_slope_gda,gradient_slope_gda, "Gradient Descent Armijo",save_dir="plots_"+name_dataset)
        
        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Gradient Descent Armijo", epochs, model_grad_desc_Arm.execution_time, Loss_gradient_descent_armijo_complete[-1], Gradient_norm_gradient_descent_armijo_complete[-1], path_img_GDA,num_sub_block=1)
        #save_results_to_excel(name_dataset,results_df_gda, file_excel)

    if method =="jacobi":
        #print("---------------------------------------------------------------")
        #print("Metodo di Jacobi")

        model_jacobi = Jacobi(num_blocks, lambda_reg, tol, max_iter,num_jobs, max_iter_in_block)
        #print(f"  Numero di blocchi:{num_blocks}\n  Lambda:{lambda_reg}\n  Tolleranza:{tol} ")
        #print("---------------------------------------------------------------")
        #print("** Applico il metodo **")
        Loss_jacobi_complete,Gradient_jacobi_complite,Gradient_norm_jacobi_complete,w_opt,iterazioni = model_jacobi.jacobi(X, y)

        #pendenza della norma del gradiente
        gradient_norm_slope_j = np.diff(Gradient_norm_jacobi_complete)

        #pendenza del vettore dei gradienti 
        gradient_slope_j = [Gradient_jacobi_complite[i] - Gradient_jacobi_complite[i - 1]  # Forza array 1D
            for i in range(1, len(Gradient_jacobi_complite))
        ]
        #print("Gradient_slope_j",gradient_slope_j)
        path_img_J=plot_loss_and_gradient(Loss_jacobi_complete,0, Gradient_norm_jacobi_complete,gradient_norm_slope_j,gradient_slope_j, "Jacobi",save_dir="plots_"+name_dataset)
        #print(Gradient_jacobi_complite)
        #print(Gradient_norm_jacobi_complete)
        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Jacobi", iterazioni, model_jacobi.execution_time, Loss_jacobi_complete[-1], Gradient_norm_jacobi_complete[-1],path_img_J,num_blocks)
        #save_results_to_excel(name_dataset,results_df_j, file_excel)

    if method =="gauss_seidel":
        #print("---------------------------------------------------------------")
        #print("Metodo di Gauss-Seidel")


        model_gauss_seidel = GaussSeidel(num_blocks, lambda_reg, tol, max_iter, max_iter_in_block,epochs)
        
        #print(f"  Numero di blocchi:{num_blocks}\n  Lambda:{lambda_reg}\n  Tolleranza:{tol} ")
        #print("---------------------------------------------------------------")
        #print("** Applico il metodo **")
        Loss_gauss_seidel_complete,Loss_gauss_seidel_block,Gradient_norm_gauss_seidel_complete,Gradient_gauss_seidel_complete,iterazioni = model_gauss_seidel.gauss_seidel(X, y)
        #print("\n** Calcolo i pesi **")
        #print("\n  -> Pesi ottimizzati con Gauss-Seidel:\n", w_opt)
        #pendenza della norma del gradiente
        gradient_norm_slope_gs = np.diff(Gradient_norm_gauss_seidel_complete)
        #pendenza del vettore dei gradienti 
        for i in range(1, len(Gradient_gauss_seidel_complete)):
                gradient_slope_gs =[Gradient_gauss_seidel_complete[i] - Gradient_gauss_seidel_complete[i - 1]]
        #print("Gradient_slope_gs",gradient_slope_gs)
        #print("\n  -> Loss con il metodo del gauss seidel:", Loss_gauss_seidel_complete)
        #print("\n  -> Loss finale con il metodo del gauss seidel:", Loss_gauss_seidel_complete[-1])
        #print("\n  -> Gradiente finale con il metodo del gauss seidel:\n", Gradient_norm_gauss_seidel_complete[-1])
        #print("\n  -> Vettore dei vettori dei gradienti con il metodo gauss seidel:\n", Gradient_gauss_seidel_complete)
        #print("\n  -> Pendenza dei vettori dei gradienti con il metodo gauss seidel:\n", gradient_slope_gs)
        #print("\n  -> Vettore delle norme dei gradienti con il metodo gauss seidel:\n", Gradient_norm_gauss_seidel_complete)

        path_img_GS=plot_loss_and_gradient(Loss_gauss_seidel_complete,Loss_gauss_seidel_block, Gradient_norm_gauss_seidel_complete,gradient_norm_slope_gs,gradient_slope_gs, "Gauss-Seidel",save_dir="plots_"+name_dataset)
      
        results_df= create_results_dataframe(joblib_num_jobs,name_dataset,Caratteristiche,Campioni,"Gauss-Seidel", iterazioni, model_gauss_seidel.execution_time, Loss_gauss_seidel_complete[-1], Gradient_norm_gauss_seidel_complete[-1],path_img_GS,num_blocks)
        #save_results_to_excel(name_dataset,results_df_gs, file_excel)
    return results_df