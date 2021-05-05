"""
Helper functions for decoding analysis
"""
import numpy as np

from dDR.dDR import dDR

import logging
log = logging.getLogger()


def parse_modelname(modelname):
    """
    Parse modelstring for decoding jobs
    """
    model_options = {
        'preprocessor': None,
        'ddr_extra': None,
        'ddr_fit': None,
        'wopt_mask': ['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL'],
        'beh_mask': ['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL']
    }
    options = modelname.split('_')
    for op in options:
        
        if op.startswith('pp'):
            #preprocessors
            model_options['preprocessor'] = op
        
        if op.startswith('dDR'):
            # dDR options
            ndim = int(op[3])
            if ndim != 2:
                model_options['ddr_extra'] = ndim - 2
            _ops = op.split('-')
            for _op in _ops:
                if _op=='allTargets':
                    model_options['ddr_fit'] = 'allTargets'
        
        if op.startswith('wopt'):
            # decoding axis fitting options
            dataset = op.split('-')[1]
            valid_trials = dataset.split('.')
            wopt_trials = []
            for v in valid_trials:
                if v=='h':
                    wopt_trials.append('HIT_TRIAL')
                elif v=='m':
                    wopt_trials.append('MISS_TRIAL')
                elif v=='f':
                    wopt_trials.append('CORRECT_REJECT_TRIAL')
                elif v=='i':
                    wopt_trials.append('INCORRECT_HIT_TRIAL')
                elif v=='p':
                    wopt_trials.append('PASSIVE_TRIALS')
            
            model_options['wopt_mask'] = wopt_trials
        
        if op.startswith('mask'):
            dataset = op.split('-')[1]
            valid_trials = dataset.split('.')
            trials = []
            for v in valid_trials:
                if v=='h':
                    wopt_trials.append('HIT_TRIAL')
                elif v=='m':
                    wopt_trials.append('MISS_TRIAL')
                elif v=='f':
                    wopt_trials.append('CORRECT_REJECT_TRIAL')
                elif v=='i':
                    wopt_trials.append('INCORRECT_HIT_TRIAL')
                elif v=='p':
                    wopt_trials.append('PASSIVE_TRIALS')
            
            model_options['beh_mask'] = trials


    return model_options


def fit_dDR(d, e1, e2, mask=None, extra_dim=None, ddr_noise=None):
    """
    d is dictionary of spike counts for difference epochs
    e1/e2 are strings specifying which data to use.
    mask is dictionary with same size as d, used to mask data prior to computing dDR axes
    """

    ddr = dDR(ddr2_init=ddr_noise, n_additional_axes=extra_dim)

    if (type(e1) is list) & (type(e2) is list):
        log.info(f"Special case -- using {e1} for Catch and {e2} for target data to define dDR axes")
        if mask is not None:
            r1 = np.concatenate([d[k][mask[k].squeeze()] for k in e1], axis=0)
            r2 = np.concatenate([d[k][mask[k].squeeze()] for k in e2], axis=0)
        else:
            r1 = np.concatenate([d[k] for k in e1], axis=0)
            r2 = np.concatenate([d[k] for k in e2], axis=0)            
        ddr.fit(r1, r2)
        return ddr
    else:
        log.info(f"Defining dDR using {e1} vs. {e2}")
        if mask is not None:
            ddr.fit(d[e1][mask[e1].squeeze()], d[e2][mask[e2].squeeze()])
        else:
            ddr.fit(d[e1], d[e2])
        return ddr


def generate_est_val(d, e1, e2, leave_one_out=False):
    """
    Generate list of est / val matrices for the pair of sounds, e1/e2.
    Returnt the list of est sets, val sets, and lok (the key (e1 or e2) of the left out sound for each item)
    """

    if leave_one_out:
        # leave out (so k trial items will be returned in each list)
        est = []
        val = []
        lok = []
        for i in range(d[e1].shape[0]):
            v = {}
            v[e1] = d[e1][i]
            v[e2] = d[e2]
            est_idx = np.array(list(set(range(d[e1].shape[0])).difference(set([i]))))
            e = {}
            e[e1] = d[e1][est_idx]
            e[e2] = d[e2]
            est.append(e)
            val.append(v)
            lok.append(e1)
        for i in range(d[e2].shape[0]):
            v = {}
            v[e2] = d[e2][i]
            v[e1] = d[e1]
            est_idx = np.array(list(set(range(d[e2].shape[0])).difference(set([i]))))
            e = {}
            e[e2] = d[e2][est_idx]
            e[e1] = d[e1]
            est.append(e)
            val.append(v)
            lok.append(e2)

        return est, val, lok

    else:
        # 50-50 split, for njacks (sort of like bootstrapping)
        raise ValueError("Not implemented")
        