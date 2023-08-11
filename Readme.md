# Deep Subtraction Angiography
### Overview
---
Unsupervised Pre-training Models for Coronary Angiogram  Subtraction and Segmentation

### Install 
---
```shell
conda create --name deepsa python==3.9
conda activate deepsa
git clone https://github.com/newfyu/DeepSA.git
cd DeepSA
pip install -r requirements.txt
```

### Run demo
---
```
python demo.py
```

### Dataset
----

- LM-CAD (Live-Mask Coronary Angiogram Dataset): 
- FS-CAD (Fine Segmentation Coronary Angiogram Dataset): 

### Train
----
Train base mode use LM-CAD dataset
```shell
python train.py
```

### Finetune
----
Finetune base mode use FM-CAD dataset
```
python finetune.py
```

### Cite
---
```
xxxx
```
