import os
import pandas as pd

input_dir = "D:/FLIR/train/RGB_adjusted"


column_names = ["filename"]

dataframe = pd.DataFrame(columns = column_names)

for f in os.listdir(input_dir):
    path = os.path.join(input_dir, f)
    path = path.replace("\\", "/")
    dataframe = dataframe.append({"filename": path }, ignore_index=True)

dataframe.to_pickle(("dataset.pkl"))