"""
Miscellaneous specific/default parameters for each batch. 
    e.g. 
        * sound offset/onset times
        * rawids per site
        * epochs / epoch names per site / batch
        * sampling rate
"""
import numpy as np

import nems_lbhb.tin_helpers as thelp

def rasterfs(batch):
    if batch==302:
        return 10
    elif batch==307:
        return 20
    elif batch==324:
        return 10
    else:
        raise ValueError(f"Batch: {batch} not supported")

def onset(batch):
    if batch==302:
        return 0.1
    elif batch==307:
        return 0.35
    elif batch==324:
        return 0.1
    else:
        raise ValueError(f"Batch: {batch} not supported")

def offset(batch):
    if batch==302:
        return 0.3
    elif batch==307:
        return 0.55
    elif batch==324:
        return 0.3
    else:
        raise ValueError(f"Batch: {batch} not supported")

def beh_mask(batch):
    if batch ==324:
        return ['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL']
    elif batch == 307:
        return ['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'MISS_TRIAL']
    elif batch == 302:
        return ['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL', 'MISS_TRIAL']
    else:
        raise ValueError(f"Batch: {batch} not supported")

def epoch_names(rec, manager, batch):
    # need to do some "hacky" stuff for batch 302 / 307 to get names to align with the TIN data
    _ra = rec.copy()
    active_epochs = [e for e in _ra.epochs.name.unique() if e.endswith('_TRIAL')]
    _ra  = _ra.and_mask(active_epochs)
    _ra = _ra.apply_mask(reset_epochs=True)

    if batch == 324:
        targets = thelp.sort_targets([f for f in _ra['resp'].epochs.name.unique() if 'TAR_' in f])
        # only keep target presented at least 5 times
        targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
        # remove "off-center targets"
        on_center = thelp.get_tar_freqs([f.strip('REM_') for f in _ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
        targets = [t for t in targets if str(on_center) in t]
        if len(targets)==0:
            # NOT ENOUGH REPS AT THIS SITE
            skip_site = True
        catch = [f for f in _ra['resp'].epochs.name.unique() if 'CAT_' in f]
        # remove off-center catches
        catch = [c for c in catch if str(on_center) in c]
        rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
        targets_str = targets
        catch_str = catch
        ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
        ref_str = ref_stim
        tar_idx = 0
    elif batch == 307:
        params = manager.get_baphy_exptparams()
        params = [p for p in params if p['BehaveObjectClass']!='Passive'][0]
        tf = params['TrialObject'][1]['TargetHandle'][1]['Names']
        targets = [f'TAR_{t}' for t in tf]
        if params['TrialObject'][1]['OverlapRefTar']=='Yes':
            snrs = params['TrialObject'][1]['RelativeTarRefdB'] 
        else:
            snrs = ['Inf']
        snrs = [s if (s!=np.inf) else 'Inf' for s in snrs]
        #catchidx = int(params['TrialObject'][1]['OverlapRefIdx'])
        refs = params['TrialObject'][1]['ReferenceHandle'][1]['Names']
        catch = ['REFERENCE'] #['STIM_'+refs[catchidx]]
        catch_str = [f'CAT_{tf[0]}+-InfdB+Noise+allREFs']
        targets_str = [f'TAR_{t}+{snr}dB+Noise' for snr, t in zip(snrs, tf)]
        targets_str = targets_str[::-1]
        targets = targets[::-1]

        # only keep targets w/ at least 5 reps in active
        targets_str = [ts for t, ts in zip(targets, targets_str) if (_ra['resp'].epochs.name==t).sum()>=5]
        targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
        
        ref_stim = [f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f]
        ref_str = [f"STIM_{tf[0]}+torc{r.split('LIN_')[1].split('_v')[0]}" for r in ref_stim]

        # only keep refs with at least 3 reps
        ref_str = [ts for t, ts in zip(ref_stim, ref_str) if (_ra['resp'].epochs.name==t).sum()>=3]
        ref_stim = [t for t in ref_stim if (_ra['resp'].epochs.name==t).sum()>=3]

        tar_idx = 0

    elif batch == 302:
        params = manager.get_baphy_exptparams()
        params = [p for p in params if p['BehaveObjectClass']!='Passive'][0]
        tf = params['TrialObject'][1]['TargetHandle'][1]['Names']
        targets = [f'TAR_{t}' for t in tf]
        pdur = params['BehaveObject'][1]['PumpDuration']
        rew = np.array(tf)[np.array(pdur)==1].tolist()
        catch = [t for t in targets if (t.split('TAR_')[1] not in rew)]
        catch_str = [(t+'+InfdB+Noise').replace('TAR_', 'CAT_') for t in targets if (t.split('TAR_')[1] not in rew)]
        targets = [t for t in targets if (t.split('TAR_')[1] in rew)]
        targets_str = [t+'+InfdB+Noise' for t in targets if (t.split('TAR_')[1] in rew)]
        ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
        ref_str = ref_stim
        tar_idx = 1

    return targets, catch, targets_str, catch_str


def which_rawids(site):
    if site == 'TAR010c':
        rawid = [123675, 123676, 123677, 123681]                                 # TAR010c
    elif site == 'AMT018a':
        rawid = [134965, 134966, 134967]                                         # AMT018a
    elif site == 'AMT020a':
        rawid = [135002, 135003, 135004]                                         # AMT020a
    elif site == 'AMT022c':
        rawid = [135055, 135056, 135057, 135058, 135059]                         # AMT022c
    elif site == 'AMT026a':
        rawid = [135176, 135178, 135179]                                         # AMT026a
    elif site == 'BRT026c':
        rawid = [129368, 129369, 129371, 129372]                                 # BRT026c
    elif site == 'BRT033b':
        rawid = [129703, 129705, 129706]                                         # BRT033b
    elif site == 'BRT034f':
        rawid = [129788, 129791, 129792, 129797, 129799, 129800, 129801]         # BRT034f
    elif site == 'BRT036b':
        rawid = [131947, 131948, 131949, 131950, 131951, 131952, 131953, 131954] # BRT036b
    elif site == 'BRT037b':
        rawid = [131988, 131989, 131990]                                         # BRT037b
    elif site == 'BRT039c':
        rawid = [132094, 132097, 132098, 132099, 132100, 132101]                 # BRT039c
    elif site == 'bbl102d':
        rawid = [130649, 130650, 130657, 130661]                                 # bbl102d
    else:
        return None

    return tuple(rawid)