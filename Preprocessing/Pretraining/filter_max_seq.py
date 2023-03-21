import pickle
import pandas as pd

data_split = sys.argv[1]
split_name = "."+data_split

if data_split == "train":
    data_split = pickle.load(open("UKBPreTrain.bencs.train", "rb"), encoding="bytes")
elif data_split == "test":
    data_split = pickle.load(open("UKBPreTrain.bencs.test", "rb"), encoding="bytes")
elif data_split == "valid":
    data_split = pickle.load(open("UKBPreTrain.bencs.valid", "rb"), encoding="bytes")

pt_list = []
pt_idsFiltered = []

for elem in valid:
    if len(elem[2]) < 3:
        pt_idsFiltered.append(elem[0])
        continue
    else:
        pt_list.append(elem)

pickle.dump(pt_list, open(f"UKBPreTrain_THREEFILTERED.bences{split_name}", 'a+b'), -1)
pickle.dump(pt_idsFiltered, open(f"UKBPreTrainTHREEFILTERED_{split_name}_removed", 'a+b'), -1)


####checking#########

# a = pickle.load(open("UKBPreTrain_THREEFILTERED.bences.train", "rb"), encoding="bytes")
# b = pickle.load(open("UKBPreTrain_THREEFILTERED.bences.test", "rb"), encoding="bytes")
# c = pickle.load(open("UKBPreTrain_THREEFILTERED.bences.valid", "rb"), encoding="bytes")

# print(a[0])

# print(len(a))
# print(len(b))
# print(len(c))

# print(len(a)+len(b)+len(c))
# print(len(a))
# max_len = 0

# for elem in a:
#     if len(elem[2]) > max_len:
#         max_len = len(elem[2])
#         print(max_len)

# max_len = 3
# i = 0
# j = 0
# k = 0

# for elem in a:
#     if len(elem[2]) < 2:
#         i += 1
# print("TRAIN_rm",i)

# for elem in b:
#     if len(elem[2]) < 2:
#         j += 1
# print("TEST_rm", j)

# for elem in c:
#     if len(elem[2]) < 2:
#         k += 1
# print("VALID_rm", k)
# print("\n")
# print("TOTAL_rm", i+j+k)
# print("NEW_TOTAL", (len(a)+len(b)+len(c)) - (i+j+k))
# print("\n")
# print("NEW_TRAIN", len(a)-i)
# print("NEW_TEST", len(b)-j)
# print("NEW_VALID", len(c)-k)
# print("\n")
# print("PERCENT REMOVED", (i+j+k)/(len(a)+len(b)+len(c)) * 100)
# print("PERCENT TRAIN REMOVED", i/len(a)*100)
# print("PERCENT TEST REMOVED", j/len(b)*100)
# print("PERCENT VALID REMOVED", k/len(c)*100)