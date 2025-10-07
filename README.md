# CoNRaD
This repository contains code and resources for the ACM WebConf 2026 submission "Content Neighborhood Ranking Distillation for Cold-Start and Long-Tail Multimodal Recommendation".


### Training
We implement CoNRaD using the [MMRec](https://github.com/enoche/MMRec) toolbox, with the same environment requirements. Where available, we use the model scripts for baseline methods from MMRec. We implement [SOIL](https://github.com/TL-UESTC/SOIL), [GUME](https://github.com/NanGongNingYi/GUME), and [TMLP](https://github.com/jessicahuang0163/TMLP) using the scripts in their respective repositories.

The model scripts for both [CoNRaD](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/models/conrad.py) and the [CoNRaD neighborhood refinement teacher](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/models/conrad_teacher.py) can be found in the [models folder](https://github.com/AnonRecSys/CoNRaD/tree/main/MMRec_CoNRaD/src/models). We note that we [modify the base MMRec trainer](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/common/trainer_conrad.py) to incorporate our learning rate decay strategy and post-training search over the $\beta$ parameter. 

Hyperparameter search ranges can be found in the [config files](https://github.com/AnonRecSys/CoNRaD/tree/main/MMRec_CoNRaD/src/configs). We also provide the [final hyperparameter settings](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/configs/CoNRaD_final_params.yml) used to produce the results in the paper.

### Data
We also mirror MMRec's data repository structure, with separate folders for each dataset. To faciliate reproducibility, we provide the post-split interaction data for the Overall, Standard, and Debiased sets for each dataset, as well as the files containing the top-ranked items by the KNN teachers for each user. We do not include the Electronics dataset due to upload limits, but will provide this via Google Drive after deanonymization.

We note that we remap the item indices from the original datasets to facilitate our cold-start analysis. Index maps for each datasets are included as the `item_lsts.pkl` file in each dataset folder. 

Item image and text features can be downloaded from the [Google Drive link](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) from MMRec. We include a [utility script](https://github.com/AnonRecSys/CoNRaD/blob/main/MMRec_CoNRaD/src/utils/conrad_feat_mapping.py) to remap these item features for use with our modified item indices.


### Cold-Start
MMRec is not set up to handle cold-start recommendation. We therefore also provide our code for this component of our work in the [Cold folder](https://github.com/AnonRecSys/CoNRaD/tree/main/Cold). The [model implementations](https://github.com/AnonRecSys/CoNRaD/tree/main/Cold/models) of our other baselines (except ALDI, which we run using its [original TensorFlow implementation](https://github.com/zfnWong/ALDI)) and the overall setup were based on the [ColdRec repository](https://github.com/YuanchenBei/ColdRec), with adjustments made to include the standard/debiased splits as well as updates to model code to improve performance and align more closely with the original implementations by the authors. 

We run CoNRaD, VBPR, and FREEDOM in MMRec, then save the predicted item content features to disk and use our cold evaluation protocol measure performance determine cold performance. We include an [example script](https://github.com/AnonRecSys/CoNRaD/blob/main/Cold/util/conrad_results.py) of this process for CoNRaD. 

We also incude the [code](https://github.com/AnonRecSys/CoNRaD/blob/main/data/data_split.py) to produce the biased/debiased/cold splits of the data.
