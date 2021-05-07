"""
Prelim look at loocv results 
    Qs: 
        Does decoding change active vs. passive?
        Does variance change active vs. passive?
        "" for performance? (Hit vs. miss vs. slow RT vs. fast RT)
"""
import matplotlib.pyplot as plt

from tools.load_results import load_loocv
from settings import ALL_SITES

modelname = 'leaveOneOut_pp-zscore_dDR2-allTargets-noise.target-mask.h+c+m+i+p_wopt-mask.h+c'
modelname = 'leaveOneOut_pp-zscore-pr_dDR2-allTargets-noise.target-mask.h+c+m+i+p_wopt-mask.h+c+m+i+p'
#modelname = 'leaveOneOut_pp-zscore_dDR2-allTargets-noise.target-mask.h+c+m+i+p_wopt-mask.h+c+m+i+p'

results = load_loocv(modelname, ALL_SITES)


# decoding of targets better in active vs passive - 
# only look at TAR, not CAT, to calculate percent correct
# keep all active trials (inc. misses, for example)
active = []
passive = []
batch = []
for site in results.site.unique():
    idx = (results.epoch_category=='TAR') & (results.discrim_category=='TAR_CAT') & (results.site==site)
    r = results[idx]
    aidx = ~r.behavior_outcome.str.contains('PASSIVE_EXPERIMENT')
    a_perf = r[aidx].correct.sum() / aidx.sum()
    p_perf = r[~aidx].correct.sum() / (aidx==False).sum()
    active.append(a_perf)
    passive.append(p_perf)

f, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.scatter(passive, active, s=25, edgecolor='k')
ax.plot([0.4, 1], [0.4, 1], 'k--')

ax.set_xlabel("Passive")
ax.set_ylabel("Active")
ax.set_title("Decoding accuracy (perc. correct)")

f.tight_layout()


# variance of data in active vs passive - 
# only look at TAR, not CAT
# keep all active trials (inc. misses, for example)
active_noise = []
passive_noise = []
active_wopt = []
passive_wopt = []
batch = []
for site in results.site.unique():
    idx = (results.epoch_category=='TAR') & (results.discrim_category=='TAR_CAT') & (results.site==site)
    r = results[idx]
    aidx = ~r.behavior_outcome.str.contains('PASSIVE_EXPERIMENT')
    nomiss = r.reaction_time != np.inf
    batch.append(np.corrcoef(r[aidx & nomiss]['reaction_time'].values.astype(np.float), (r[aidx & nomiss]['val_noise_projection'].values.astype(np.float)))[0, 1])
    a_noise = r[aidx].val_noise_projection.std()
    p_noise = r[~aidx].val_noise_projection.std()
    a_wopt = r[aidx].val_wopt_projection.std()
    p_wopt = r[~aidx].val_wopt_projection.std()
    active_noise.append(a_noise)
    passive_noise.append(p_noise)
    active_wopt.append(a_wopt)
    passive_wopt.append(p_wopt)

f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(passive_noise, active_noise, s=25, edgecolor='k')
ax[0].plot([0, 5], [0, 5], 'k--')
ax[0].set_xlabel("Passive")
ax[0].set_ylabel("Active")
ax[0].set_title("Variance on noise axis")

ax[1].scatter(passive_wopt, active_wopt, s=25, edgecolor='k')
ax[1].plot([0, 5], [0, 5], 'k--')
ax[1].set_xlabel("Passive")
ax[1].set_ylabel("Active")
ax[1].set_title("Variance on decoding axis")

f.tight_layout()

# does variance on these two axes change between HIT / MISS?
cor_noise = []
inc_noise = []
cor_wopt = []
inc_wopt = []
for site in results.site.unique():
    idx = (results.epoch_category=='TAR') & (results.discrim_category=='TAR_CAT') & (results.site==site)
    r = results[idx]
    aidx = ~r.behavior_outcome.str.contains('PASSIVE_EXPERIMENT') & (r.behavior_outcome.str.contains('HIT_TRIAL') | r.behavior_outcome.str.contains('CORRECT_REJECT_TRIAL'))
    pidx = ~r.behavior_outcome.str.contains('PASSIVE_EXPERIMENT') & (r.behavior_outcome.str.contains('MISS_TRIAL') | r.behavior_outcome.str.contains('INCORRECT_HIT_TRIAL'))
    if sum(pidx) > 10:
        a_noise = r[aidx].val_noise_projection.std()
        p_noise = r[pidx].val_noise_projection.std()
        a_wopt = r[aidx].val_wopt_projection.std()
        p_wopt = r[pidx].val_wopt_projection.std()
        cor_noise.append(a_noise)
        inc_noise.append(p_noise)
        cor_wopt.append(a_wopt)
        inc_wopt.append(p_wopt)

f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(inc_noise, cor_noise, s=25, edgecolor='k')
ax[0].plot([0, 5], [0, 5], 'k--')
ax[0].set_xlabel("Incorrect")
ax[0].set_ylabel("Correct")
ax[0].set_title("Variance on noise axis")

ax[1].scatter(inc_wopt, cor_wopt, s=25, edgecolor='k')
ax[1].plot([0, 5], [0, 5], 'k--')
ax[1].set_xlabel("Incorrect")
ax[1].set_ylabel("Correct")
ax[1].set_title("Variance on decoding axis")

f.tight_layout()

# does variance on these two axes change for HI vs. LO reaction time?
cor_noise = []
inc_noise = []
cor_wopt = []
inc_wopt = []
for site in results.site.unique():
    idx = (results.epoch_category=='TAR') & (results.discrim_category=='TAR_CAT') & (results.site==site)
    r = results[idx]
    idx = ~r.behavior_outcome.str.contains('PASSIVE_EXPERIMENT') & (r.behavior_outcome.str.contains('HIT_TRIAL'))
    r = r[idx]
    rts = r.reaction_time
    aidx = (r.reaction_time.values <= np.median(rts))
    pidx = (r.reaction_time.values > np.median(rts))
    if sum(pidx) > 10:
        a_noise = r[aidx].val_noise_projection.std()
        p_noise = r[pidx].val_noise_projection.std()
        a_wopt = r[aidx].val_wopt_projection.std()
        p_wopt = r[pidx].val_wopt_projection.std()
        cor_noise.append(a_noise)
        inc_noise.append(p_noise)
        cor_wopt.append(a_wopt)
        inc_wopt.append(p_wopt)

f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(inc_noise, cor_noise, s=25, edgecolor='k')
ax[0].plot([0, 5], [0, 5], 'k--')
ax[0].set_xlabel("Slow RT")
ax[0].set_ylabel("Fast RT")
ax[0].set_title("Variance on noise axis")

ax[1].scatter(inc_wopt, cor_wopt, s=25, edgecolor='k')
ax[1].plot([0, 5], [0, 5], 'k--')
ax[1].set_xlabel("Slow RT")
ax[1].set_ylabel("Fast RT")
ax[1].set_title("Variance on decoding axis")

f.tight_layout()

plt.show()

