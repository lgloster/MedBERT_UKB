import pandas as pd
import pickle

##Convert ICD10 to 'types' dictionary###

codings = pd.read_csv("./coding19_mod.csv")
types = {'empty_pad':0}

for elem in codings['coding']:
    if elem not in types:
        types[elem] = max(types.values())+1
pickle.dump(types, open('coding19.types', 'wb'), -1)

# f = pickle.load(open("coding19.types", "rb"), encoding="bytes")
# print(f)