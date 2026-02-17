import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from regressions import RegressionModel


def run_regression(model : RegressionModel, data) : 
 
    x = get_list_of_variable_explain(data)
    y = get_explained_variable(data)
    
    min_alpha, max_alpha = get_intiutive_alpha_range(model, x, y)
    run_train_and_test(model, x, y , min_alpha , max_alpha)


   
def get_list_of_variable_explain(data):
    selected_columns = [
    'MIN', 'FGM', '3PM' ,'FTM', 'FT%' ,'REB' ,'AST', 'STL', 'BLK', 'TO'
    ]
    x = data[selected_columns]
    #x = data.drop(columns = "salary")
    return x

def get_explained_variable(data):
    y = np.log(data['salary'])
    return y


#initiutive 
def get_intiutive_alpha_range(model : RegressionModel, x : pd.DataFrame, y):
    list_alphas = get_list_of_alpha(7,-4,100)
    coefs = []
    
    for alph in list_alphas:
        model.fit(x, y, alph)
        coefs.append(model.get_coefs())
    
    show_graph_of_weight(list_alphas, coefs)

    min_alpha = float(input("Enter the Minimum Alpha : "))
    max_alpha = float(input("Enter the Maximum Alpha : "))
    
    return min_alpha, max_alpha


#train & test 
def run_train_and_test(model : RegressionModel, x : pd.DataFrame, y , min_alpha : int, max_alpha : int):

    X_train, X_test , y_train, y_test = model_selection.train_test_split(scale(x), y, test_size=0.5, random_state=1)
    list_alphas = get_list_of_alpha(max_alpha, min_alpha ,15)

    alpha_and_mse = {}
    alpha_and_coefs = {}

    for alph in list_alphas:
        model.fit(X_train, y_train, alph)
        alpha_and_coefs[alph] = model.get_coefs()
        result_test_run = model.predict(X_test) 
        mse =  mean_squared_error(y_test, result_test_run)
        alpha_and_mse[alph] = mse

    print_result(alpha_and_mse, alpha_and_coefs)
    return alpha_and_coefs



def get_list_of_alpha(max_alpha, min_alpha, num_sample):
    list_alphas = 10**np.linspace(max_alpha, min_alpha, num_sample)*0.5
    #list_alphas = np.logspace(min_alpha, max_alpha, num_sample) * 0.5
    return list_alphas


def show_graph_of_weight(list_alphas : list, coefs : list):
    ax = plt.gca()
    ax.plot(list_alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.show()


def print_result(alpha_and_mse, alpha_and_coefs): 
    for alpha in alpha_and_mse.keys():
        mse = alpha_and_mse[alpha]
        coefs = alpha_and_coefs[alpha]
        print(f"Alpha: {alpha:.5f}")
        print(f"MSE: {mse:.5f}")
        print(f"Coefficients: {coefs}")
        print("-" * 50)  

