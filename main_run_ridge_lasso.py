from regressions import RidgeRegression, LassoRegression
from run import run_regression
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns



def delete_row_in_col_by_name(df, col_name):
    return df.drop(columns = col_name)

def load_your_data():
    df =   pd.read_csv("UTF-8NBA_Data.csv")
    return df.dropna()

def clean_data(data : pd.DataFrame):
    X = data.drop(columns=["salary"])  
    y = data["salary"]   
    return X, y 



def main():
    col_name = ["player", "team", "POS"]
    data = load_your_data()
    data = delete_row_in_col_by_name(data, col_name)

    """ !! choose what model you want to run !! """
    model =  RidgeRegression()
    #model = LassoRegression()
    run_regression(model, data)

if __name__ == "__main__":
    main()
