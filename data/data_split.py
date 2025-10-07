import pandas as pd
import random
import os
import pickle


def warm_cold_split(df_orig, cold_item_prop, dataset):
    df = df_orig.copy(deep=True)
    items = set(df["item"].values)
    num_warm_users = 0
    while num_warm_users < len(set(df["user"].values)):
        df = df_orig.copy(deep=True)
        cold_items = random.sample(list(items), int(len(items) * cold_item_prop))

        warm_items = sorted(items.difference(cold_items))
        num_warm = len(warm_items)

        warm_item_idx_map = {v: i for i, v in enumerate(warm_items)}
        cold_item_idx_map = {v: i + num_warm for i, v in enumerate(sorted(cold_items))}
        idx_map = {**warm_item_idx_map, **cold_item_idx_map}

        df["item"] = df["item"].apply(lambda x: idx_map[x])

        val_cold_items = [
            cold_item_idx_map[i] for i in cold_items[: len(cold_items) // 2]
        ]
        test_cold_items = [
            cold_item_idx_map[i] for i in cold_items[len(cold_items) // 2 :]
        ]

        val_cold_df = df[df["item"].isin(val_cold_items)]
        test_cold_df = df[df["item"].isin(test_cold_items)]
        warm_df = df[df["item"] < num_warm]
        num_warm_users = len(set(warm_df["user"].values))
    print("cold sampled")
    all_items = warm_items + sorted(cold_items)

    with open("./data/%s/item_lsts.pkl" % dataset, "wb") as f:
        pickle.dump((all_items, warm_items, cold_items), f)
    return warm_df, val_cold_df, test_cold_df


def get_train_core(df, core_size):
    item_users = df.groupby("item")["user"].apply(list).to_dict()
    train_core = []
    remaining = []
    for i, i_users in item_users.items():
        random.shuffle(i_users)
        train_core += [(u, i) for u in i_users[:core_size]]
        remaining += [(u, i) for u in i_users[core_size:]]

    train_core_df = pd.DataFrame(train_core, columns=["user", "item"])
    train_core_user_items = train_core_df.groupby("user")["item"].apply(list).to_dict()

    remaining_df = pd.DataFrame(remaining, columns=["user", "item"])
    remaining_user_items = remaining_df.groupby("user")["item"].apply(list).to_dict()
    all_users = set(df["user"].values)

    final_core = []
    for u in all_users:
        if u in train_core_user_items:
            u_core_items = train_core_user_items[u]
            if len(u_core_items) < core_size:
                try:
                    u_core_items += random.sample(
                        remaining_user_items[u],
                        min(
                            core_size - len(u_core_items), len(remaining_user_items[u])
                        ),
                    )
                except:
                    pass
        else:
            u_core_items = random.sample(
                remaining_user_items[u], min(core_size, len(remaining_user_items[u]))
            )
        final_core += [(u, i) for i in u_core_items]
    final_core_set = set(final_core)
    final_remaining = set(remaining).difference(final_core_set)
    return pd.DataFrame(final_core_set, columns=["user", "item"]), pd.DataFrame(
        final_remaining, columns=["user", "item"]
    )


def get_final_splits(
    df,
    core_size=2,
    test_deb_size=0.1,
    test_standard_size=0.1,
    val_deb_size=0.1,
    val_standard_size=0.1,
):
    orig_df_size = df.shape[0]
    train_core, remaining = get_train_core(df, core_size)

    deb_total = test_deb_size + val_deb_size
    remaining_item_probs = (1 / remaining["item"].value_counts()).to_dict()
    remaining["item_prob"] = remaining["item"].apply(lambda x: remaining_item_probs[x])
    deb_df = remaining.sample(
        n=int(deb_total * orig_df_size), weights="item_prob"
    ).drop("item_prob", axis=1)
    standard_df = remaining.drop(deb_df.index, axis=0).drop("item_prob", axis=1)

    deb_val = deb_df.sample(frac=val_deb_size / deb_total)
    deb_test = deb_df.drop(deb_val.index, axis=0)

    standard_val = standard_df.sample(n=int(val_standard_size * orig_df_size))
    standard_test = standard_df.drop(standard_val.index, axis=0).sample(
        n=int(test_standard_size * orig_df_size)
    )
    standard_train = standard_df.drop(
        pd.concat([standard_val, standard_test]).index, axis=0
    )
    final_train = pd.concat([train_core, standard_train])
    return final_train, deb_val, standard_val, deb_test, standard_test


def create_mmrec_out(dataset):

    train = pd.read_csv("./data/%s/warm_train.csv" % dataset)
    train["x_label"] = 0
    val = pd.read_csv("./data/%s/overall_val.csv" % dataset)
    val["x_label"] = 1

    test = pd.read_csv("./data/%s/overall_test.csv" % dataset)
    test["x_label"] = 2
    warm = pd.concat([train, val, test])
    warm.columns = ["userID", "itemID", "x_label"]
    warm["timestamp"] = None
    warm["rating"] = None
    warm = warm[["userID", "itemID", "rating", "timestamp", "x_label"]]
    warm.to_csv(
        "../MMRec_CoNRaD/data/%s_cold_deb/%s_cold_deb.inter" % (dataset, dataset),
        sep="\t",
        index=False,
    )


def split(
    dataset,
    core_size=2,
    test_deb_size=0.1,
    test_standard_size=0.1,
    val_deb_size=0.1,
    val_standard_size=0.1,
    cold_item_prop=0.15,
):
    df = pd.read_csv(
        "./data/%s/%s.csv" % (dataset, dataset)
    )
    warm_df, val_cold_df, test_cold_df = warm_cold_split(df, cold_item_prop, dataset)

    final_train, deb_val, standard_val, deb_test, standard_test = get_final_splits(
        warm_df,
        core_size,
        test_deb_size,
        test_standard_size,
        val_deb_size,
        val_standard_size,
    )

    pth = "./data/%s" % dataset
    if not os.path.exists(pth):
        os.makedirs(pth)
    final_train.to_csv(pth + "/warm_train.csv", index=False)
    deb_val.to_csv(pth + "/deb_val.csv", index=False)
    standard_val.to_csv(pth + "/standard_val.csv", index=False)
    deb_test.to_csv(pth + "/deb_test.csv", index=False)
    standard_test.to_csv(pth + "/standard_test.csv", index=False)
    val_cold_df.to_csv(pth + "/cold_val.csv", index=False)
    test_cold_df.to_csv(pth + "/cold_test.csv", index=False)
    pd.concat([deb_test, standard_test]).to_csv(pth + "/overall_test.csv", index=False)
    pd.concat([deb_val, standard_val]).to_csv(pth + "/overall_val.csv", index=False)
    create_mmrec_out(dataset)


for dataset in ["baby", "clothing", "sports", "electronics"]:
    split(dataset)
