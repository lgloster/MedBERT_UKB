import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

###Original modification of UKB data file###

dataFile = pd.read_csv(".../Merged_ICD10_sequential.txt", sep="\t")
idFile = pd.read_csv(".../Merged_ICD10_sequential_ids.txt", sep="\t")

dataFile = dataFile.drop(columns=["CR_fromICD9", "CR_ICD10","DR_ICD10","HO_ICD10","HO_fromICD9","SR_ICD10","GP_fromread"])
idFile = idFile.drop(columns=["CR_fromICD9", "CR_ICD10","DR_ICD10","HO_ICD10","HO_fromICD9","SR_ICD10","GP_fromread"])


dataFile.insert(0, "UKB_id", "")

start_idx = idFile["start_idx"].tolist()
end_idx = idFile["end_idx"].tolist()
UKB_ids = []

for idx in tqdm(range(len(idFile.index))):
    id_list = [idFile.iloc[idx]["UKB ID"] for x in range(start_idx[idx], end_idx[idx])]
    UKB_ids.extend(id_list)

dataFile["UKB_id"] = UKB_ids

# #this is raw data with codes in list form merged with UKB_ids (no dummy ids/expanded codes)
# dataFile.to_csv("Merged_ICD10_All.csv", encoding='utf-8', header=True, index=False)

dataFile = pd.read_csv("Merged_ICD10_All.csv", sep=',')

dataFile["Codes"]=dataFile["Codes"].str.split(',')
dataFile = dataFile.sort_values(by=['UKB_id', 'Age'], ascending=True)
dataFile = dataFile.reset_index(drop=True)
dataFile.insert(0, "pt_id", "")

pt_ids = []


for Pt, group in enumerate(dataFile.groupby("UKB_id")):
    
    id_list = [Pt for x in range(len(group[1]))]
    pt_ids.extend(id_list)
    
dataFile["pt_id"] = pt_ids
dataFile = dataFile.explode("Codes")
print(dataFile)

dataFile.to_csv("Merged_ICD10_for_pretrain.csv", encoding='utf-8', header=True, index=False)
