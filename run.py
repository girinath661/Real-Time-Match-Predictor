# importing the requrired libraries

import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import time


if __name__ == "__main__":

    # creating the object to pass the arguments
    parser = argparse.ArgumentParser()

    # adding argument 1 to the script
    parser.add_argument("--weights_path",type=str,required=True,help="pass the weights path")
    
    # adding argument 2 to the script
    parser.add_argument("--data_path",type=str,required=False,default="./data.csv",help = "pass the data path")
    
    # adding argument 3 to the script
    parser.add_argument('--num_preds',type=int,required=True,help='how many predictions you need ?')
    
    # declaring the arguments
    args = parser.parse_args()

    weights_path = args.weights_path # adding weights path to the variable
    data_path = args.data_path # adding data path to the variable
    num_preds = args.num_preds # adding number of predictions to the num_preds 


    # Loading the model
    model = tensorflow.keras.models.load_model(weights_path)
    
    # Loading the data
    data = pd.read_csv(data_path)

    # Dropping the unwanted columns
    data.drop(columns=["Timestamp","Email Address"],axis=1,inplace=True)
    
    # Encoding the Data
    for column in data.columns:
        le = le.fit(data[column])
        data[column] = le.fit_transform(data[column])
    # print(data.shape)
    
    counter = 0
    var = 0
    for row in range(len(data)):
        if counter == num_preds:
            break
        observation = data.loc[row]   # Extracting each row
        observation = np.array(observation) # converting each row into numpy array
        observation = observation.reshape(1,10) # reshaping the row into (1,10)
        start = time.time()
        model.predict(observation) # making the prediction
        end = time.time()
        total = end - start # calculating the time
        var += total
        counter +=1
    
    # calculating the total time for num_preds provided
    if num_preds ==1:
        print(f"total time taken for {num_preds} prediction is {var:.4f} seconds")
    else:
        print(f"total time taken for {num_preds} predictions is {var:.4f} seconds")
    
    
    
  
    
    