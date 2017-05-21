import tensorflow as tf 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split


def load_data():
    """
    Loading the data
    X_train - training input data
    X_valid - validate input data
    y_train - corresponding training output data
    y_valid - corresponding validate output data
    """
    data_df = pd.read_csv('Ch2_001/final_example.csv')
    extension = pd.Series()
    X = data_df[['frame_id']].values

    image_X = pd.DataFrame()

    #Add .jpg to image names
    for x in X:
        image_X = image_X.append([str(x[0])+'.jpg'])

    y = data_df['steering_angle'].values
    
    X_train, X_valid, y_train, y_valid = train_test_split(image_X,y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid



if __name__ == '__main__':
    load_data()