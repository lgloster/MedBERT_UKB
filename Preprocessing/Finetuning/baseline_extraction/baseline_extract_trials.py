import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle

#get baseline UKB dataset
baseline_path = ".../MI_baseline.csv"
baseline = pd.read_csv(baseline_path, sep=",")

#trial index paths
trials_Feb = ".../trial_idx_Myocardial_Infarction_av1+10_02.20.23.npz"
trials_Oct = ".../trial_idx_Myocardial_Infarction_av1+10_10.28.22.npz"

#preparing labels
labels = pd.read_csv(".../Myocardial_Infarction_labels.txt",sep='\t')
labels = labels[["UKB ID","time_window_incident/nan/10.0/nan/av1"]].dropna()
labels.reset_index(inplace=True,drop=True)

num_trials=10
idx_dict = {}
def read_trial_idx(trial_num, idx_file):

    idx_file = np.load(idx_file)

    train_idx = idx_file["train_trial{}".format(trial_num)]
    valid_idx = idx_file["valid_trial{}".format(trial_num)]
    test_idx = idx_file["test_trial{}".format(trial_num)]

    return train_idx, valid_idx, test_idx

for trial in range(1, num_trials+1):
    train_idx, valid_idx, test_idx = read_trial_idx(trial, idx_file=trials_Feb)
    idx_dict[f"Trial_{trial}"] = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx
    }


for key,val in idx_dict.items():
    for idx, array in val.items():

        #get correct indices from Myocardial labels file
        idx_list = labels[labels.index.isin(array)]
       
        #get cases and controls into lists
        cases = idx_list[idx_list["time_window_incident/nan/10.0/nan/av1"] == 1] 
        case_list = list(cases["UKB ID"].values)

        controls = idx_list[idx_list["time_window_incident/nan/10.0/nan/av1"] == 0]
        control_list = list(controls["UKB ID"].values) 

        #create list of UKB IDs
        idx_listed = list(idx_list["UKB ID"].values)

        #filter UKB data by IDs from labels file
        ukb_pts = baseline[baseline["UKB_id"].isin(idx_listed)]
        
        #split codes list into tuple, sort by ascending ID and Age
        ukb_pts["Codes"]=ukb_pts["Codes"].str.split(',')
        ukb_pts = ukb_pts.sort_values(by=['UKB_id', 'Age'], ascending=True)
        ukb_pts = ukb_pts.reset_index(drop=True)

        #add true labels
        ukb_pts.loc[ukb_pts["UKB_id"].isin(case_list), "MI_Label"] = 1 
        ukb_pts.loc[ukb_pts["UKB_id"].isin(control_list), "MI_Label"] = 0 

        #initialize dummy ID column
        ukb_pts.insert(0, "pt_id", "")

        pt_ids = []

        for Pt, group in enumerate(ukb_pts.groupby("UKB_id")):
            
            id_list = [Pt for x in range(len(group[1]))]
            pt_ids.extend(id_list)
            
        ukb_pts["pt_id"] = pt_ids
        ukb_pts = ukb_pts.explode("Codes")


        print("DONE", key, idx)
        print(key, idx,"#"*10, len(idx_listed), "PTS", len(ukb_pts["UKB_id"].unique()))
        ukb_pts.to_csv(f"{key}_{idx}.csv", sep=',', index=False, header=True)



