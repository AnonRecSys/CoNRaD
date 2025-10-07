import numpy as np
import pickle


def remap(dataset):
    _, warm_items, _ = pickle.load(
        open("./data/%s_conrad/item_lsts.pkl" % dataset, "rb")
    )
    text_feats_mapped = np.load("./data/%s/text_feat.npy" % (dataset + "_cold_deb"))[
        warm_items
    ]
    image_feats_mapped = np.load("./data/%s/image_feat.npy" % (dataset + "_cold_deb"))[
        warm_items
    ]

    with open("./data/%s_conrad/text_feat_mapped.npy" % dataset, "wb") as f:
        np.save(f, text_feats_mapped)
    with open("./data/%s_conrad/image_feat_mapped.npy" % dataset, "wb") as f:
        np.save(f, image_feats_mapped)


for dataset in ["baby", "clothing", "electronics", "sports"]:
    remap(dataset)
    print(dataset)
