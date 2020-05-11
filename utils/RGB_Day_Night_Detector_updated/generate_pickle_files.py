import pandas as pd
import os


dataset_dir = "D:/KAIST_Thermal_Dataset_copy"

day_set = ["set00", "set01", "set02"]

night_set = ["set03", "set04", "set05"]

column_names = ["filename", "label"]

dataframe = pd.DataFrame(columns = column_names)

#day is labeled as 0

for day_dir in day_set:
    sub_dir = os.path.join(dataset_dir, day_dir, "images", day_dir)
    for sub_sub_dir in os.listdir(sub_dir):
        if not sub_sub_dir.startswith('.'):

            sub_sub_sub_dir = os.path.join(sub_dir,sub_sub_dir, "visible")
            for file in os.listdir(sub_sub_sub_dir):
                if not file.startswith('.'):
                    full_path = os.path.join(sub_sub_sub_dir, file)
                    dataframe = dataframe.append({"filename": full_path, "label": 0}, ignore_index=True)

#night is labeled as 1
for night_dir in night_set:
    sub_dir = os.path.join(dataset_dir, night_dir, "images", night_dir)
    for sub_sub_dir in os.listdir(sub_dir):
        if not sub_sub_dir.startswith('.'):
            sub_sub_sub_dir = os.path.join(sub_dir,sub_sub_dir, "visible")
            for file in os.listdir(sub_sub_sub_dir):
                if not file.startswith('.'):
                    full_path = os.path.join(sub_sub_sub_dir, file)
                    dataframe = dataframe.append({"filename": full_path, "label": 1}, ignore_index=True)


dataframe = dataframe.sample(frac=1)
# dataframe = dataframe[:100]
dataframe = dataframe.reset_index(drop=True)
dataframe.to_pickle("dataset.pkl")









