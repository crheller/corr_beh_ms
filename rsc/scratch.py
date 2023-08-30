"""
Load rsc results and do preliminary checks:
    active vs. passive diff (overall and by batch)
    relationship with behavior (overall and by batch)
    results for different time windows (by batch / overall where possible)
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

rsc_file = "/auto/users/hellerc/results/corr_beh_ms/rsc/rsc_df.pickle"

df = pd.read_pickle(rsc_file)
df = df[df.area=="A1"]
df = df[df.snr != -np.inf]
# for each batch, plot active vs. passive
# early timewindow
f, ax = plt.subplots(1, 3, figsize=(9, 3))

for i, b in enumerate(np.unique(df["batch"])):
    b_df = df[(df.batch==b)]
    if b in [302, 324]:
        b_df = b_df[b_df.tbin=="0_0.1"]
    else:
        b_df = b_df[b_df.tbin=="0.05_0.15"]
    bysite = b_df.groupby(by="site").mean()
    ax[i].plot(bysite[["active", "passive"]].to_numpy().T, "k", alpha=0.5)
    ax[i].set_title(f"batch: {int(b)}")

for a in ax:
    a.set_ylim((-0.05, 0.25))
    a.set_ylabel("rsc")
    a.set_xlabel("State")
    a.set_xlim((-1, 2))
    a.set_xticks([0, 1])
    a.set_xticklabels(["active", "passive"])
f.suptitle("Early window (~0 - 0.1 seconds)")
f.tight_layout()

# middle time window
f, ax = plt.subplots(1, 3, figsize=(9, 3))

for i, b in enumerate(np.unique(df["batch"])):
    b_df = df[(df.batch==b)]
    if b in [302, 324]:
        b_df = b_df[b_df.tbin=="0.1_0.2"]
    else:
        b_df = b_df[b_df.tbin=="0.15_0.25"]
    bysite = b_df.groupby(by="site").mean()
    ax[i].plot(bysite[["active", "passive"]].to_numpy().T, "k", alpha=0.5)
    ax[i].set_title(f"batch: {int(b)}")

for a in ax:
    a.set_ylim((-0.05, 0.25))
    a.set_ylabel("rsc")
    a.set_xlabel("State")
    a.set_xlim((-1, 2))
    a.set_xticks([0, 1])
    a.set_xticklabels(["active", "passive"])
f.suptitle("Middle window (~0.1 - 0.2 seconds)")
f.tight_layout()

# middle time window
f, ax = plt.subplots(1, 3, figsize=(9, 3))

for i, b in enumerate(np.unique(df["batch"])):
    b_df = df[(df.batch==b)]
    if b in [302, 324]:
        b_df = b_df[b_df.tbin=="0.2_0.3"]
    else:
        b_df = b_df[b_df.tbin=="0.25_0.35"]
    bysite = b_df.groupby(by="site").mean()
    ax[i].plot(bysite[["active", "passive"]].to_numpy().T, "k", alpha=0.5)
    ax[i].set_title(f"batch: {int(b)}")

for a in ax:
    a.set_ylim((-0.05, 0.25))
    a.set_ylabel("rsc")
    a.set_xlabel("State")
    a.set_xlim((-1, 2))
    a.set_xticks([0, 1])
    a.set_xticklabels(["active", "passive"])
f.suptitle("Late window (~0.2 - 0.3 seconds)")
f.tight_layout()

# ======================
# correlation of delta rsc with behavior
beh_met1 = "DIref"
beh_met2 = "DIref"
groupby = ["site", "snr", "f"]
batches = [302, 307, 324, "all"]
# early time window
f, ax = plt.subplots(1, 4, figsize=(12, 3.5))

for i, b in enumerate(batches):
    if b != "all":
        b_df = df[(df.batch==b)]
        if b in [302, 324]:
            b_df = b_df[b_df.tbin=="0_0.1"]
        else:
            b_df = b_df[b_df.tbin=="0.05_0.15"]
        beh_metric = beh_met1
        if b in [302, 307]:
            beh_metric = beh_met2
        bysite = b_df.groupby(by=groupby).mean()
        drsc = bysite["passive"] - bysite["active"]
        perf = bysite[beh_metric]
        vv = perf.isna()==False
        perf = perf[vv]; drsc = drsc[vv]
    else:
        mask = ((df.batch==307) & (df.tbin=="0.05_0.15")) | (df.batch.isin([302, 324]) & (df.tbin=="0_0.1"))
        b_df = df.iloc[mask.values]
        bysite = b_df.groupby(by=groupby).mean()

        drsc1 = bysite[bysite.batch.isin([302, 307])]["passive"] - bysite[bysite.batch.isin([302, 307])]["active"] 
        drsc2 = bysite[bysite.batch.isin([324])]["passive"] - bysite[bysite.batch.isin([324])]["active"] 
        perf1 = bysite[bysite.batch.isin([302, 307])][beh_met2]
        perf2 = bysite[bysite.batch.isin([324])][beh_met1]

        drsc = np.concatenate([drsc1, drsc2])
        perf = np.concatenate([perf1.values, perf2.values])
        vv = np.isnan(perf)==False
        perf = perf[vv]; drsc = drsc[vv]
    

    ax[i].scatter(perf, drsc, c="k", s=25, edgecolor="none")
    cc, pval = ss.pearsonr(perf, drsc)
    cc = np.round(cc, 3)
    pval = np.round(pval, 3)
    try:
        ax[i].set_title(f"batch: {int(b)}\ncc: {cc}, pval: {pval}")
    except:
        ax[i].set_title(f"all batches\ncc: {cc}, pval: {pval}")

for a in ax:
    # a.set_ylim((-0.05, 0.25))
    a.set_ylabel("rsc")
    a.set_xlabel("Performance")
f.suptitle("Early window (~0.0 - 0.1 seconds)")
f.tight_layout()

# middle time window
f, ax = plt.subplots(1, 4, figsize=(12, 3.5))

for i, b in enumerate(batches):
    if b != "all":
        b_df = df[(df.batch==b)]
        if b in [302, 324]:
            b_df = b_df[b_df.tbin=="0.1_0.2"]
        else:
            b_df = b_df[b_df.tbin=="0.15_0.25"]
        beh_metric = beh_met1
        if b in [302, 307]:
            beh_metric = beh_met2
        bysite = b_df.groupby(by=groupby).mean()
        drsc = bysite["passive"] - bysite["active"]
        perf = bysite[beh_metric]
        vv = perf.isna()==False
        perf = perf[vv]; drsc = drsc[vv]
    else:
        mask = ((df.batch==307) & (df.tbin=="0.15_0.25")) | (df.batch.isin([302, 324]) & (df.tbin=="0.1_0.2"))
        b_df = df.iloc[mask.values]
        bysite = b_df.groupby(by=groupby).mean()

        drsc1 = bysite[bysite.batch.isin([302, 307])]["passive"] - bysite[bysite.batch.isin([302, 307])]["active"] 
        drsc2 = bysite[bysite.batch.isin([324])]["passive"] - bysite[bysite.batch.isin([324])]["active"] 
        perf1 = bysite[bysite.batch.isin([302, 307])][beh_met2]
        perf2 = bysite[bysite.batch.isin([324])][beh_met1]

        drsc = np.concatenate([drsc1, drsc2])
        perf = np.concatenate([perf1.values, perf2.values])
        vv = np.isnan(perf)==False
        perf = perf[vv]; drsc = drsc[vv]
    

    ax[i].scatter(perf, drsc, c="k", s=25, edgecolor="none")
    cc, pval = ss.pearsonr(perf, drsc)
    cc = np.round(cc, 3)
    pval = np.round(pval, 3)
    try:
        ax[i].set_title(f"batch: {int(b)}\ncc: {cc}, pval: {pval}")
    except:
        ax[i].set_title(f"all batches\ncc: {cc}, pval: {pval}")

for a in ax:
    # a.set_ylim((-0.05, 0.25))
    a.set_ylabel("rsc")
    a.set_xlabel("Performance")
f.suptitle("Middle window (~0.1 - 0.2 seconds)")
f.tight_layout()

# late time window
f, ax = plt.subplots(1, 4, figsize=(12, 3.5))

for i, b in enumerate(batches):
    if b != "all":
        b_df = df[(df.batch==b)]
        if b in [302, 324]:
            b_df = b_df[b_df.tbin=="0.2_0.3"]
        else:
            b_df = b_df[b_df.tbin=="0.25_0.35"]
        beh_metric = beh_met1
        if b in [302, 307]:
            beh_metric = beh_met2
        bysite = b_df.groupby(by=groupby).mean()
        drsc = bysite["passive"] - bysite["active"]
        perf = bysite[beh_metric]
        vv = perf.isna()==False
        perf = perf[vv]; drsc = drsc[vv]
    else:
        mask = ((df.batch==307) & (df.tbin=="0.25_0.35")) | (df.batch.isin([302, 324]) & (df.tbin=="0.2_0.3"))
        b_df = df.iloc[mask.values]
        bysite = b_df.groupby(by=groupby).mean()

        drsc1 = bysite[bysite.batch.isin([302, 307])]["passive"] - bysite[bysite.batch.isin([302, 307])]["active"] 
        drsc2 = bysite[bysite.batch.isin([324])]["passive"] - bysite[bysite.batch.isin([324])]["active"] 
        perf1 = bysite[bysite.batch.isin([302, 307])][beh_met2]
        perf2 = bysite[bysite.batch.isin([324])][beh_met1]

        drsc = np.concatenate([drsc1, drsc2])
        perf = np.concatenate([perf1.values, perf2.values])
        vv = np.isnan(perf)==False
        perf = perf[vv]; drsc = drsc[vv]
    

    ax[i].scatter(perf, drsc, c="k", s=25, edgecolor="none")
    cc, pval = ss.pearsonr(perf, drsc)
    cc = np.round(cc, 3)
    pval = np.round(pval, 3)
    try:
        ax[i].set_title(f"batch: {int(b)}\ncc: {cc}, pval: {pval}")
    except:
        ax[i].set_title(f"all batches\ncc: {cc}, pval: {pval}")

for a in ax:
    # a.set_ylim((-0.05, 0.25))
    a.set_ylabel("rsc")
    a.set_xlabel("Performance")
f.suptitle("Late window (~0.2 - 0.3 seconds)")
f.tight_layout()