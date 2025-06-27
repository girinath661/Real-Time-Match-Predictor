step1: create a environment with python=3.12 ---> conda create -n predictions python==3.12

step2: activate the environment --> conda activate predictions

step3: install the requirements --> pip install -r requirements.txt

step4: python run.py --weights_path "./weights.h5" --data_path "./data.csv" --num_preds 5

NOTE: if you have any custom weight file trained on the same data provided change the path while passing --weights_path