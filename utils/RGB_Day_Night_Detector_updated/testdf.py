import pandas as pd
import os


column_names = ["filename", "label"]

dataframe = pd.DataFrame(columns = column_names)

directory = "D:/KAIST_Dataset/val/images/visible"


for file in os.listdir(directory):
    file_num = int(file.split(".")[0])
    full_file = os.path.join(directory, file)
    if file_num < 33411:
        dataframe = dataframe.append({"filename": full_file, "label": 0}, ignore_index=True)
    else:
        dataframe = dataframe.append({"filename": full_file, "label": 1}, ignore_index=True)

dataframe.to_pickle("test.pkl")