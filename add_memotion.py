import pandas as pd
import os
import shutil
from mmf.utils.configuration import Configuration

# Requirements
# 1. Download memotion from kaggle. It should be downloaded as archive.zip
# https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.kaggle.com%2Fwilliamscott701%2Fmemotion-dataset-7k
# 2. Extract archive.zip into the root folder of mmf 
# 3. Ensure you have already run mmf_convert_hm --zip_file=data.zip --password=password --bypass_checksum=1 to genereate the initial hateful_memes dir
# 5. Ensure you already have the label_memotion.jsonl in the root folder as well  
# 4. Run this python script from mmf root folder
# To run with original training set, use mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes  dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/train.jsonl
# To run with original training set + memotion, use mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes  dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/train_with_memotion.jsonl
# Citation : Cleaned memotion dataset & labelling is from the  HateDetectron submission. Flow of adding the memotion dataset to the existing hateful_meme dataset is also based on their workflow, which can be found in this link  https://colab.research.google.com/drive/1O0m0j9_NBInzdo3K04jD19IyOhBR1I8i?usp=sharing#scrollTo=EgtqtKPGpKQb

print("Reading label_memotion.jsonl")
labeled_memo_samples = pd.read_json("./label_memotion.jsonl", lines=True)['img']
memotion_image_labels = [i.split('/')[1] for i in list(labeled_memo_samples)]
img_dir = "./memotion_dataset_7k/images/"

# Get directory of existing data directory where the original hateful_memes is set
configuration = Configuration()
config = configuration.get_config()
data_dir = config.env.data_dir
hateful_memes_img_dir = data_dir + "/datasets/hateful_memes/defaults/images/img"
print("Dataset location of original hateful memes is ", hateful_memes_img_dir)

# # # # Copy over memotion dataset images in original hateful_memes dataset
# for img in memotion_image_labels:
#     shutil.copy(f"{img_dir+img}", os.path.join(hateful_memes_img_dir, f"{img}"))
#     print("Copied ", img , "into hateful_memes dataset directory")

# print("Reading original hateful memes training set info: train.jsonl ")
# train = pd.read_json(os.path.join(data_dir, "datasets/hateful_memes/defaults/annotations/train.jsonl"), lines=True)
# # Load labeled Memotion data
# memotion = pd.read_json("./label_memotion.jsonl", lines=True)
# train = pd.concat([train, memotion], axis=0)
# # Shuffle data
# train = train.sample(frac=1).reset_index(drop=True)

# # Write new jsonl file
# train_json = train.to_json(orient='records', lines=True)

# with open(os.path.join(data_dir, "datasets/hateful_memes/defaults/annotations/train_with_memotion.jsonl"), "w", encoding='utf-8') as memotion_file:
#     memotion_file.write(train_json)
#     print("Created train_with_memotion.jsonl at ",os.path.join(data_dir, "datasets/hateful_memes/defaults/annotations/train_with_memotion.jsonl") )
