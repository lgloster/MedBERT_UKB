# lrasmy @ Zhilab 2019/08/10
# This script processes Cerner dataset and builds pickled lists including a full list that includes all patient encounters information 
# The output data are (c)pickled, and suitable for training of BERT_EHR models 
# Usage: feed this script with patient targets file that include patient_id, encounter_id and other relevant labels field ( here we use mortality, LOS and time to readmit) 
# and the main data fields like diagnosis, procedures, medication, ...etc and if you decide to use a predefined vocab file (tokenization/ types dict)
# additionally you can specify sample size , splitting to train,valid,and test sets and the output file path
# So you can run as follow
# python preprocess_pretrain_data.py <data_File>  <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <data_size>
# Output files:
# <output file>.types: Python dictionary that maps string diagnosis codes to integer indexes
# <output file>.ptencs: List of pts_encs_data
# <output file>.encs: slimmer list that only include tokenized encounter data and labels
# <output file>.bencs: slimmer list that only include tokenized encounter data and labels, along with other list representing segments (visits)
# The above files will also be splitted to train,validation and Test subsets using the Ratio of 7:1:2


import sys
from optparse import OptionParser

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import os
#pd.options.mode.chained_assignment = None 
#import timeit


### Random split to train ,test and validation sets
# def split_fn(pts_ls,pts_sls,outFile):
#    print ("Splitting")
#    dataSize = len(pts_ls)
#    np.random.seed(0)
#    ind = np.random.permutation(dataSize)
#    nTest = int(0.2 * dataSize)
#    nValid = int(0.1 * dataSize)
#    test_indices = ind[:nTest]
#    valid_indices = ind[nTest:nTest+nValid]
#    train_indices = ind[nTest+nValid:]

#    for subset in ['train','valid','test']:
#        if subset =='train':
#             indices = train_indices
#        elif subset =='valid':
#             indices = valid_indices
#        elif subset =='test':
#             indices = test_indices
#        else: 
#             print ('error')
#             break
#        subset_ptencs = [pts_ls[i] for i in indices]
#        subset_ptencs_s =[pts_sls[i] for i in indices]
#        ptencsfile = outFile +'.ptencs.'+subset
#        bertencsfile = outFile +'.bencs.'+subset
#        pickle.dump(subset_ptencs, open(ptencsfile, 'a+b'), -1)
#        pickle.dump(subset_ptencs_s, open(bertencsfile, 'a+b'), -1)
       
### Main Function
if __name__ == '__main__':
    
   #targetFile= sys.argv[1]
#   diagFile= sys.argv[1]
  typeFile= sys.argv[1]
#   outFile = sys.argv[3]
  p_samplesize = int(sys.argv[2]) ### replace with argparse later

#   parser = OptionParser()
#   (options, args) = parser.parse_args()
 
   
  #_start = timeit.timeit()
   
  debug=False
  #np.random.seed(1)

  curr_dir = os.getcwd()

  #iterate over directory
  
  for filename in os.listdir(curr_dir):
    
    
    name_split = filename.split("_")
    
    
    if name_split[0] == "Trial":
        print(name_split)
        outFile = "MI_"+name_split[0] + "_" + name_split[1]
        subset = name_split[2]
        
       
  ## Data Loading
        print (" data file" )   
        #   data_diag= pd.read_csv(diagFile, sep='\t')
        data_diag= pd.read_csv(filename, sep=',')     
        #   data_diag.columns=['pt_id','admit_dt_tm','discharge_dt_tm', 'diagnosis', 'poa', 'diagnosis_priority','third_party_ind']
        data_diag.columns=['pt_id','UKB_id','Age','Codes', "MI_Label"]

        if typeFile=='NA': 
            types={'empty_pad':0}
        else:
            with open(typeFile, 'rb') as t2:
                    types=pickle.load(t2)
                    
                

        #### Sampling
        
        if p_samplesize>0:
            print ('Sampling')
            ptsk_list=data_diag['pt_id'].drop_duplicates()
            pt_list_samp=ptsk_list.sample(n=p_samplesize)
            n_data= data_diag[data_diag["pt_id"].isin(pt_list_samp.values.tolist())]  
        else:
            n_data=data_diag
        #n_data.admit_dt_tm.fillna(n_data.discharge_dt_tm, inplace=True) ##, checked the data and no need for that line

            
        ##### Data pre-processing
        print ('Start Data Preprocessing !!!')
        print("DATA SIZE:", n_data)
        count=0
        pts_ls=[]
        pts_sls=[]
        # max_freq = 0
        
        for Pt, group in n_data.groupby('pt_id'):
            
            # group["letter"] = group["Codes"].str.get(0)
            # max_freq = round(max(group['letter'].value_counts(normalize=True)), 2)
            condition_label = group["MI_Label"].unique()
            condition_label = int(condition_label[0])
            
            
            pt_encs=[]
            #   time_tonext=[]
            #   pt_los=[]
            full_seq=[]
            v_seg=[]
            #   pt_discdt=[]
            #   pt_addt=[]
            pt_ls=[]
            v=0
            for Age,subgroup in group.sort_values(['UKB_id','Age','Codes'], ascending=True).groupby('Age', sort=False): ### changing the sort order
                    
                v=v+1
                diag_l=np.array(subgroup['Codes'].drop_duplicates()).tolist()
                

                if len(diag_l)> 0:
                    diag_lm=[]
                    for code in diag_l: 
                        if code in types:
                            diag_lm.append(types[code])
                        else: 
                            types[code] = max(types.values())+1
                            diag_lm.append(types[code])
                        
                        v_seg.append(v)
                    
                    full_seq.extend(diag_lm)
                    

                enc_l=[diag_l,diag_lm]
                pt_encs.append(enc_l)
        
            if len(pt_encs)>0:
                pt_ls.append(pt_encs)
        
            pts_ls.append(pt_ls)
            pts_sls.append([Pt,condition_label,full_seq,v_seg])
            
            
        
            count=count+1

            if count % 1000 == 0: print ('processed %d pts' % count)
            
            # if count % 100000 == 0:
            #     print ('dumping %d pts' % count)
            #     split_fn(pts_ls,pts_sls,outFile)
            #     pts_ls=[]
            #     pts_sls=[]
                    
                
        # split_fn(pts_ls,pts_sls,outFile)
        print("FINISHED", outFile)
        ptencsfile = outFile +'.ptencs.'+subset
        bertencsfile = outFile +'.bencs.'+subset
        pickle.dump(pts_ls, open(ptencsfile, 'a+b'), -1)
        pickle.dump(pts_sls, open(bertencsfile, 'a+b'), -1)   
        # pickle.dump(types, open(outFile+'.types', 'wb'), -1)  
