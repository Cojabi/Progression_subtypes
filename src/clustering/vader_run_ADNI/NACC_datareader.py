import pandas as pd
import numpy as np
import os
import glob
from vader.hp_opt.interface.abstract_data_reader import AbstractDataReader

class DataReader(AbstractDataReader):
    features = [feat.split(".")[0] for feat in glob.glob("NACC_tables/*_ADNI_norm.csv")]
    time_points = ["0", "12", "24", "36"]
    time_point_meaning = "month"
    ids_list = None

    def read_data(self, directory: str) -> np.ndarray:
        times = [0, 12, 24, 36]
        #read all csv files in path into one list of pd.DataFrames
        files = glob.glob(f"{directory}/*_ADNI_norm.csv")
        files.sort()
        csvs = [pd.read_csv(csv, index_col=0) for csv in files]
        tensor = np.stack(csvs, axis=2)
        return tensor[:,0:4,:]
