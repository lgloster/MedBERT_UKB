import pandas as pd
import numpy as np
from numpy import nan
from tqdm import tqdm
from datetime import timedelta

###opening files
BASELINE_PATH = ".../df_53_visit_date.txt"
BIRTHYEAR_PATH = ".../df_34_birthyear.txt"
RAW_PATH = ".../Merged_ICD10_sequential.txt"
RAW_LABELS = ".../Merged_ICD10_sequential_ids.txt"

###for finetuning
MI_labels = pd.read_csv(".../Myocardial_Infarction_labels.txt", sep="\t", header=0)
MI_labels = MI_labels[["UKB ID","time_window_incident/nan/10.0/nan/av1"]].dropna()
MI_labels.reset_index(drop=True,inplace=True) #LG 03/17/2023

#load dataframes
baseFile = pd.read_csv(BASELINE_PATH, sep='\t')
birthFile = pd.read_csv(BIRTHYEAR_PATH, sep='\t')
codesFile = pd.read_csv(RAW_PATH, sep='\t')
labelFile = pd.read_csv(RAW_LABELS, sep='\t')

#find age of first visit to UKB
baseFile = baseFile.drop(columns=["f.53.2.0", "f.53.1.0"])
baseFile = baseFile.dropna()
birthFile = birthFile.dropna()

year_merge = baseFile.merge(birthFile, left_on="f.eid", right_on="f.eid")

year_merge["f.53.0.0"] = pd.to_datetime(year_merge["f.53.0.0"]).dt.year
year_merge["first_visit_age"] = year_merge["f.53.0.0"]-year_merge["f.34.0.0"]

year_merge["first_visit_age"] = year_merge["first_visit_age"].astype(np.int64)
year_merge["f.eid"] = year_merge["f.eid"].astype(np.int64)

#dropping info source columns
codesFile = codesFile.drop(columns=["CR_fromICD9", "CR_ICD10","DR_ICD10","HO_ICD10","HO_fromICD9","SR_ICD10","GP_fromread"])
labelFile = labelFile.drop(columns=["CR_fromICD9", "CR_ICD10","DR_ICD10","HO_ICD10","HO_fromICD9","SR_ICD10","GP_fromread"])

#initialize lists for new codesFile columns
labelFile["idx_list"] = labelFile.values.tolist()
id_list = []


#extract ids
for elem in tqdm(labelFile["idx_list"]):
    for e in range(elem[1], elem[2]):
        id_list.append(elem[0])
        
codesFile["UKB_id"] = id_list

#reset index
codesFile = codesFile.reset_index(drop=True)
year_merge = year_merge.reset_index(drop=True)

#drop unnecessary columns
year_merge = year_merge.rename(columns={"f.eid":"UKB_id"})
year_merge = year_merge.drop(columns=["f.53.0.0","f.34.0.0"])

#merge birth and codes files on UKB_id
code_year_merge = codesFile.merge(year_merge, how='inner', on="UKB_id")

#filter by baseline Age criteria
code_year_merge = code_year_merge[code_year_merge["Age"] <= code_year_merge["first_visit_age"]]
code_year_merge = code_year_merge.drop(columns=["first_visit_age"])

#cases
case_list = MI_labels[MI_labels["time_window_incident/nan/10.0/nan/av1"] == 1]
case_list = list(case_list["UKB ID"].values)

#controls
control_list = MI_labels[MI_labels["time_window_incident/nan/10.0/nan/av1"] == 0]
control_list = list(control_list["UKB ID"].values)

#extract case data from labels and add label column
df_cases = code_year_merge[code_year_merge["UKB_id"].isin(case_list)]
df_controls = code_year_merge[code_year_merge["UKB_id"].isin(control_list)]
df_cases["MI_label"] = 1
df_controls["MI_label"] = 0

#merge cases and controls
df_all = pd.concat([df_cases, df_controls])

#sort be ascending UKB_ids
df_all = df_all.sort_values(by=["UKB_id"])

#split codes lists and make new row for each single code instance per patient
df_all["Codes"] = df_all["Codes"].str.split(",")
df_all = df_all.explode("Codes")

df_all.to_csv(".../MI_baseline.csv", sep=",", header=True, index=False)
