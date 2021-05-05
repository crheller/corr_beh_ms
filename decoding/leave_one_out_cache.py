"""
Use leave-one-out cross validation to measure the decoding accuracy and noise on single trials (and a percent correct over all trials).

Decoding Procedure:
    1) ID sound pair (e.g. target 1 vs. catch)
    2) Generate leave-one-out sets (there will be k target trial number of sets)
    3) Project data into dDR space (dU/noise computed for specific stim pair, or computed over all trials / targets - depends on flag set. Try both)
        * Note - maybe this isn't the best idea (dDR sort of overfit), but it allows us to look at how the noise changes on per trial basis in an 
            identical space (i.e. how the projection into the noise space changes - does it get tighter along decoding axis, or not?).
    4) Calculate decoding axis in test space / define decision boundary
        * Which trials the decoding axis is caculated on depends on the modelname - 
            could compute over all trials, just HIT trials, just MISS, HIT+MISS+FA etc.
    5) Project left out point onto axis and determine if correct

Results structure:
    * DF per-stimulus pair per target trial (for each stim pair, there will be k trial results)
        * save result (correct / incorrect neural decoding)
        * save value of left out projection (for overall dprime calculation)
        * save dDR axes (fixed across trials, but not necessarily across stim pairs)
        * save evecs / evals in dDR space
        * save wopt (decoding axis)
        * save value of projection of left out trial onto the *noise* axis per trial
        * save behavioral outcome (RT / HIT / MISS / FA)
    * Extras
        * Number of cells at the site. 
        * Number of trials per epoch / behavior condition?

Example modelnames:
    leaveOneOut_pp-zscore-pr_dDR2-allTargets_wopt-h.m.f
        - preprocessing: zscore and regress pupil, 2-D dDR, computed using all targets, optimal decoder measured using hit/miss/FA trials
    
    leaveOneOut_pp-center-mask.h.p_dDR3-thisTarget_wopt-h.f
        - preprocessing: center data, mask hits/passive trials, 3-D dDR, computed using just a single target, optimal decoder measured using hit/FA trials
"""
from itertools import combinations
import numpy as np 

from tools.loaders import load_data
from dDR.dDR import dDR
import tools.decoding_helpers as dh

from charlieTools.nat_sounds_ms.decoding import compute_dprime

import logging
log = logging.getLogger()

modelname = 'leaveOneOut_pp-zscore-pr_dDR2-allTargets_wopt-h.m.f'

# TODO - option for selecting which data goes into the dDR algorithm
# TODO - option for specifying a fixed noise axis for dDR

# 1) Parse modelname options
model_options = dh.parse_modelname(modelname)
log.info(f"Parsed modelname options: {model_options}")


# 2) Load and preprocess the data
site = 'TAR010c'
batch = 307
data = load_data(site=site, batch=batch, 
                            preprocessor=model_options['preprocessor'], 
                            beh_mask=model_options['beh_mask'])


# 3) Perform decoding analysis(es) / save results
# get sound pairs
sound_pairs = list(combinations(data['catch']+data['targets'], 2))

# do decoding
for sp in sound_pairs:

    # define dDR space over all trials (don't want space to change for each left out point)
    if model_options['ddr_fit'] == 'allTargets': 
        # define dDR by grouping all targets as single class vs. all catches as single class (so, this doesn't *have* to happen inside the loop --
        # little bit redundant, but easier to read)
        ddr = dh.fit_dDR(data['d'], data['catch'], data['targets'], 
                                extra_dim=model_options['ddr_extra']
                                )
    else:
        # define dDR specifically for this sound pair
        ddr = dh.fit_dDR(data['d'], sp[0], sp[1], 
                                extra_dim=model_options['ddr_extra']
                                )

    # for each (left out) trial, fit decoding axis on remaining data, project left out data, and compute statistics
    c = []
    for (e1, e2) in [[sp[0], sp[1]], [sp[1], sp[0]]]:
        for i in range(data['d'][e1].shape[0]):
            val1 = data['d'][e1][i]
            val2 = data['d'][e2]
            est_idx = np.array(list(set(range(data['d'][e1].shape[0])).difference(set([i]))))
            est1 = data['d'][e1][est_idx]
            est2 = data['d'][e2]

            # project data onto dDR axes
            val1 = ddr.transform(val1)
            est1, est2 = (ddr.transform(est1), ddr.transform(est2))

            # fit decoding axis
            mask = data['TRIAL_masks'][model_options['wopt_mask'][0]]
            for k in model_options['wopt_mask']:
                for k2 in mask.keys():
                    mask[k2] = mask[k2] | data['TRIAL_masks'][k][k2]
            
            # compute decoding axis on the est set
            A = est1[mask[e1][est_idx].squeeze()].T
            B = est2[mask[e2].squeeze()].T
            dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(A, B)

            # project the left out point
            n_wopt = wopt / np.linalg.norm(wopt)
            val = val1.dot(n_wopt)[0]

            # which is it closer to?
            g1 = A.T.dot(n_wopt).mean(); g2 = B.T.dot(n_wopt).mean()

            # things to save -- 
            #   * save result (correct / incorrect neural decoding)
            #   * save value of left out projection (for overall dprime calculation)
            #   * save dDR axes (fixed across trials, but not necessarily across stim pairs)
            #   * save evecs / evals in dDR space
            #   * save wopt (decoding axis)
            #   * save value of projection of left out trial onto the *noise* axis per trial
            #   * save value of projection of left out trial onto the *decoding* axis per trial
            #   * save behavioral outcome (RT / HIT / MISS / FA)

            if abs(val-g1) < abs(val-g2):
                # correct
                c.append(1)
            else:
                # incorrect
                c.append(0)