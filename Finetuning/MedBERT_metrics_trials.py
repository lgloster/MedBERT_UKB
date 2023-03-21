from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import json
import sys


class MedBERT_report():
    def __init__(self, epoch=None):
        self.metrics = metrics
        self.epoch = epoch

        print("REPORT CLASS TRIGGERED HERE"+"-"*30, self.epoch, sep='\n')
    
    def track_counts(self, t_labels, t_preds,v_labels, v_preds):
        class_counts={}

        #get counts of train/valid labels
        tl_counts = np.unique(t_labels,return_counts=True)
        vl_counts = np.unique(v_labels, return_counts=True)
        

        #get metrics from roc_curve
        t_fpr, t_tpr, t_threshold = metrics.roc_curve(t_labels, t_preds)
        v_fpr, v_tpr, v_threshold = metrics.roc_curve(v_labels, v_preds)

        #find best decision boundary
        t_threshold = sorted(list(zip(np.abs(t_tpr - t_fpr), t_threshold)), key=lambda i: i[0], reverse=True)[0][1]
        v_threshold = sorted(list(zip(np.abs(v_tpr - v_fpr), v_threshold)), key=lambda i: i[0], reverse=True)[0][1]

        #make prediction class decisions
        t_bin_pred = [1 if i >= t_threshold else 0 for i in t_preds]
        v_bin_pred = [1 if i >= v_threshold else 0 for i in v_preds]

        #count predictions
        tp_counts = np.unique(t_bin_pred,return_counts=True)
        vp_counts = np.unique(v_bin_pred,return_counts=True)
        

        #Instantiate dictionary structure for count metrics
        counts_list = {"valid_labels": vl_counts, "valid_preds": vp_counts, "train_labels": tl_counts, "train_preds": tp_counts}
        class_counts.update({"Counts":{"train": {"labels": {},"preds": {}},"valid": {"labels": {},"preds": {}}}})

        #populate dictionary with counts (exceptions for when only ONE CLASS is predicted instead of two)
        for key, val in counts_list.items():
            print(key, val)
            key_split = key.split("_")
            try:
                class_counts["Counts"][f"{key_split[0]}"][f"{key_split[1]}"][f"class_{round(val[0][0])}"] = val[1][0]
                class_counts["Counts"][f"{key_split[0]}"][f"{key_split[1]}"][f"class_{round(val[0][1])}"] = val[1][1]
                class_counts["Counts"][f"{key_split[0]}"][f"{key_split[1]}"]["total_count"] = val[1][0] + val[1][1]
            except:
                try:
                    class_counts["Counts"][f"{key_split[0]}"][f"{key_split[1]}"][f"class_{round(val[0][0])}"] = val[1][0]
                    class_counts["Counts"][f"{key_split[0]}"][f"{key_split[1]}"]["total_count"] = val[1][0]
                except:
                    print("EXCEPTION", key,val, val[0], val[1])

        return class_counts

            
    def classification_report(self, t_labels, t_preds, v_labels, v_preds, train_loss, valid_loss):

        #get metrics from roc_curve
        t_fpr, t_tpr, t_threshold = metrics.roc_curve(t_labels, t_preds)
        v_fpr, v_tpr, v_threshold = metrics.roc_curve(v_labels, v_preds)

        #find best decision boundary
        t_threshold = sorted(list(zip(np.abs(t_tpr - t_fpr), t_threshold)), key=lambda i: i[0], reverse=True)[0][1]
        v_threshold = sorted(list(zip(np.abs(v_tpr - v_fpr), v_threshold)), key=lambda i: i[0], reverse=True)[0][1]

        #make prediction class decisions
        t_bin_pred = [1 if i >= t_threshold else 0 for i in t_preds]
        v_bin_pred = [1 if i >= v_threshold else 0 for i in v_preds]
        
        #get AUROC score
        t_auc = metrics.roc_auc_score(t_labels, t_preds)
        v_auc = metrics.roc_auc_score(v_labels, v_preds)

        #make report
        t_report = metrics.classification_report(t_labels, t_bin_pred, target_names=["class_0", "class_1"], output_dict=True)
        v_report = metrics.classification_report(v_labels, v_bin_pred, target_names=["class_0", "class_1"], output_dict=True)

        #Matthews correlation coefficient(MCC)
        t_mcc = matthews_corrcoef(t_labels, t_bin_pred)
        v_mcc = matthews_corrcoef(v_labels, v_bin_pred)

        #Confusion matrix (CM)
        t_cm = confusion_matrix(t_labels, t_bin_pred, labels = [0,1])
        v_cm = confusion_matrix(v_labels, v_bin_pred, labels=[0,1])
        
        #adding auc/loss to final report dict
        t_report["auroc"] = t_auc
        t_report["loss"] = train_loss
        t_report["mcc"] = t_mcc

        v_report["auroc"] = v_auc
        v_report["loss"] = valid_loss
        v_report["mcc"] = v_mcc

        #adding roc metrics to list (position of tuples indicated by order of "train"/"valid")
        roc_file_list = ["train_fpr", "train_tpr", "train_threshold", "valid_fpr", "valid_tpr", "train_confmatrx", "valid_confmatrx"]
        roc_file = ((list(t_fpr), list(t_tpr), t_threshold), (list(v_fpr), list(v_tpr), v_threshold), t_cm, v_cm, roc_file_list)

        return t_report, v_report, roc_file

    #creates CSV with run metrics for one Trial report AFTER training
    def avg_metrics(self, report_json, json_name):

        #instantiate dataframe
        df = pd.DataFrame(columns = ["Fold","Epoch","Train/Val","Precision", "Recall", "F1_Score", "Accuracy", "AUC", "MCC", "Loss"])
        
        #open json
        with open(report_json, 'r') as f:
            report = json.load(f)
        
        #looping through JSON to collect metrics for train/val
        a = []
        for fold, epoch in report.items():
            print(fold)
            l = []
            f_split = fold.split("_")
            for e, item in epoch.items():
                k = report[fold][e]["Counts"].keys()
                e_split = e.split(" ")
                for loop in k:
                    if loop == "train":
                        l = []
                        l.append(f_split[1])
                        l.append(e_split[1])
                        l.append(loop)
                        l.append(report[fold][e]["Counts"]["train"]["train_report"]["weighted avg"]["precision"])
                        l.append(report[fold][e]["Counts"]["train"]["train_report"]["weighted avg"]["recall"])
                        l.append(report[fold][e]["Counts"]["train"]["train_report"]["weighted avg"]["f1-score"])
                        l.append(report[fold][e]["Counts"]["train"]["train_report"]["accuracy"])
                        l.append(report[fold][e]["Counts"]["train"]["train_report"]["auroc"])
                        l.append(report[fold][e]["Counts"]["train"]["train_report"]["mcc"])
                        l.append(report[fold][e]["Counts"]["train"]["train_loss"])
                        a.append(l)
                
                    elif loop == "valid":
                        l = []
                        l.append(f_split[1])
                        l.append(e_split[1])
                        l.append(loop)
                        l.append(report[fold][e]["Counts"]["valid"]["valid_report"]["weighted avg"]["precision"])
                        l.append(report[fold][e]["Counts"]["valid"]["valid_report"]["weighted avg"]["recall"])
                        l.append(report[fold][e]["Counts"]["valid"]["valid_report"]["weighted avg"]["f1-score"])
                        l.append(report[fold][e]["Counts"]["valid"]["valid_report"]["accuracy"])
                        l.append(report[fold][e]["Counts"]["valid"]["valid_report"]["auroc"])
                        l.append(report[fold][e]["Counts"]["valid"]["valid_report"]["mcc"])
                        l.append(report[fold][e]["Counts"]["valid"]["valid_loss"])
                        a.append(l)
                
                
        #populate dataframe and save as CSV
        for elem in a:
            # print(elem)
            df.loc[len(df)] = elem
        print(df)
        df.to_csv(f".../{json_name}.csv", sep=",", index=False, header=True)
            
    
#if running standalone AFTER training: provide arg for json file PATH  
if __name__ == "__main__":
    #filepath
    load_json = sys.argv[1]

    #getting file name
    json_name = load_json.split("/")
    json_name = json_name[-1]
    json_name = json_name.split(".")
    json_name = json_name[0]

    print("REPORT: ",json_name)
    MedBERT_report().avg_metrics(load_json, json_name)
