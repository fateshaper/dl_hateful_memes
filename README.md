# OMSCS 7643 Fall 2023 Project

This is our OMSCS 7643 Group Project, on the Meta's Hateful Meme Challenge when using Multimodal Deep Learning.

Authors:

* ylow33@gatech.edu
* gcheang3@gatech.edu
* jchin47@gatech.edu

## Proposal / Overview


In 2020, Meta released the [Hateful Meme Challenge](https://mmf.sh/docs/challenges/hateful_memes_challenge/). Along with this with the [mmf](https://github.com/facebookresearch/mmf) framework.

* We wanted to add additional data from [Memotion]( https://competitions.codalab.org/competitions/20629) and see whether it affects the performance of the mutlimodal models. 
* We also explored using image & text embeddings with [open_clip](https://github.com/mlfoundations/open_clip)

## Installation

We are using Ubuntu 22.04.2 LTS (GNU/Linux 6.2.0-37-generic x86_64), along with a Nvidia RTX 4090 Card.

For background information on installation on various python / cuda drivers, please refer to this [blog post](https://lowyx.com/posts/deep-learning-rig/).


The main instructions can be found at the [mmf instlalation docs](https://mmf.sh/docs/) but a couple of updates are required for us to get it working. Based on the [latest pytorch getting started](https://pytorch.org/get-started/locally/), it requires Python 3.8. **Hence, do not follow the mmf's documentation to install python 3.7**. If you are reading this and PyTorch requires python 3.9 and above, please adjust accordingly. 

```
conda create -n mmf python=3.8
conda activate mmf
```

```bash
git clone https://github.com/facebookresearch/mmf.git
cd mmf
```

In `requirements.txt`, please edit out `pycocotools`. This is because it will cause the insallation to break.

```
pip install --editable .
```

```bash
pip3 install torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu118
```

Then, based on the pytorch version you have, you will also need to update it by following [TorchText](https://github.com/pytorch/text). Since we are using Pytorch 2.1.0, we require torch text `0.16.0`.

```bash
pip install torchtext==0.16.0
```

After this, you should run the above command again


```bash
pip3 install torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu118
```

Finally, to check everything is working as intended: 

```python
import torch
torch.cuda.is_available()
torch.tensor([1.0, 2.0]).cuda()
```

To install [open_clip](https://github.com/mlfoundations/open_clip) (please refer to link for latest instructions)

```bash
pip install open_clip_torch
```

## Running models

Running models can be found at the documentation found [here](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes).

Here are some examples with our CLIP text embeddings:

```bash
mmf_run config=projects/hateful_memes/configs/open_clip_text_encoding/defaults.yaml model=open_clip_text_encoding dataset=hateful_memes
```

For example if we want to include the Memotion dataset:

```bash
mmf_run config=projects/hateful_memes/configs/open_clip_text_encoding/defaults.yaml model=open_clip_text_encoding dataset=hateful_memes dataset_config.hateful_memes.annotations.train[0]=hateful_memes/defaults/annotations/train_with_memotion.jsonl dataset_config.hateful_memes.features.train[0]=hateful_memes/defaults/feature_test/detectron.lmdb
```

To get test/validation results after you have created your model artifact:


```bash
mmf_run config=projects/hateful_memes/configs/open_clip_text_encoding/defaults.yaml model=open_clip_text_encoding dataset=hateful_memes run_type=test checkpoint.resume_file=save/open_clip_text_encoding_final.pth checkpoint.resume_pretrained=False
```

## Adding Memotion data set

<TODO>