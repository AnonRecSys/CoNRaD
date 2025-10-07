# CoNRaD
This repository contains code and resources for the ACM WebConf 2026 submission "Content Neighborhood Ranking Distillation for Cold-Start and Long-Tail Multimodal Recommendation". 

### Training
We implement CoNRaD using the [MMRec](https://github.com/enoche/MMRec) toolbox. 

The model scripts for both [CoNRaD](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/models/conrad.py) and the [CoNRaD neighborhood refinement teacher](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/models/conrad_teacher.py) can be found in the [models folder](https://github.com/AnonRecSys/CoNRaD/tree/main/MMRec_CoNRaD/src/models). We note that we [modify the base MMRec trainer](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/common/trainer_conrad.py) to incorporate our learning rate decay strategy and post-training search over the $\beta$ parameter. 

Hyperparameter information, including relevant search ranges, can be found in the [config files](https://github.com/AnonRecSys/CoNRaD/tree/main/MMRec_CoNRaD/src/configs).

### Data
We also mirror MMRec's data repository structure, with separate folders for each dataset. To faciliate reproducibility, we provide the post-split interaction data for the Overall, Standard, and Debiased sets for each dataset, as well as the files containing the top-ranked items by the KNN teachers for each user. We do not include the Electronics dataset due to upload limits, but will provide this via Google Drive after deanonymization.

We note that we remap the item indices from the original datasets to facilitate our cold-start analysis. Index maps for each datasets are included as the `item_lsts.pkl` file in each dataset folder. 

Item image and text features can be downloaded from the [Google Drive link](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) from MMRec. We include a [utility script](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/utils/conrad_feat_mapping.py) to remap these item features for use with our modified item indices.
