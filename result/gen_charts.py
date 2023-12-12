import re
import numpy as np
import matplotlib.pyplot as plt

filename = "train_2023_12_12T02_34_45.log"
model_name = "OpenCLIP_memotion"

# "train_2023_12_10T14_33_05.log" = "MMBT"
# "train_2023_12_10T15_48_32.log" = "MMBT_memotion"
# "train_2023_12_10T10_01_41.log" = "Visualbert"
# "train_2023_12_10T12_06_37.log" = "Visualbert_memotion"
# "train_2023_12_10T19_29_52.log" = "BERT_OpenCLIP"
# "train_2023_12_11T00_12_34.log" = "BERT_OpenCLIP_memotion"
# "train_2023_12_11T23_28_27.log" = "OpenCLIP"
# "train_2023_12_12T02_34_45.log" = "OpenCLIP_memotion"


f = open(filename, "r")
lines = f.readlines()

train_data = np.zeros((220, 2))
val_data = np.zeros((22, 4))
train_insert_counter = 0
val_insert_counter = 0

for i in range(len(lines)):
    if "train/hateful_memes" in lines[i]:
        start_index = lines[i].find("progress")
        data = re.findall(r'\d+\.*\d*', lines[i][start_index:])
        train_data[train_insert_counter, :] = [data[0], data[3]]
        train_insert_counter += 1
    elif "val/hateful_memes" in lines[i]:
        start_index = lines[i].find("progress")
        data = re.findall(r'\d+\.*\d*', lines[i][start_index:])
        val_data[val_insert_counter, :] = [data[0], data[2], data[4], data[7]]
        val_insert_counter += 1


plt.plot(train_data[:, 0], train_data[:, 1])
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy Loss")
plt.title(f"{model_name} Training Curve")
plt.savefig(f"{model_name}_train_loss")
plt.clf()
plt.plot(val_data[:, 0], val_data[:, 1])
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy Loss")
plt.title(f"{model_name} Validation Curve")
plt.savefig(f"{model_name}_val_loss")
plt.clf()
plt.plot(val_data[:, 0], val_data[:, 2])
plt.plot(val_data[:, 0], val_data[:, 3])
plt.xlabel("Iterations")
plt.ylabel("Accuracy/AUROC")
plt.title(f"{model_name} Validation Metrics")
plt.legend(["Accuracy", "AUROC"])
plt.savefig(f"{model_name}_val_metrics")
plt.clf()

print(val_data)