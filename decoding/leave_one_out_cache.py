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
    leaveOneOut_pp-zscore.pr_dDR2-allTargets_wopt-h.m.f
        - preprocessing: zscore and regress pupil, 2-D dDR, computed using all targets, optimal decoder measured using hit/miss/FA trials
    
    leaveOneOut_pp-center_dDR3-thisTarget_wopt-h.f
        - preprocessing: center data, 3-D dDR, computed using just a single target, optimal decoder measured using hit/FA trials
"""

from tools.loaders import load_data

# Load and preprocess the data
site = 'TAR010c'
batch = 307
data = load_data(site=site, batch=batch, preprocessor='pp-zscore')

# Perform decoding analysis / save results