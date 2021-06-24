'''
Load behavior data. Options depend on the batch, since each task is slightly different.
'''
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import parse_cellid

import tools.batch_params as bp
import tools.preprocessors as pp

import numpy as np
import logging
log = logging.getLogger()

def load_data(site, batch, preprocessor=None, beh_mask=None, dict_ops=None, recache=False, **options):
    """
    Load data for site / batch.
        - Options will get filled to defaults here, but you can override them if you'd like by passing
            them into the function (for this purporse, probably don't want to do that though - want things
            to be standardized across analyses)
        - Same with dict_ops - these specify how to collapse / extract responses for each epoch into spike count matrices
        - preprocessor is a decoding modelkey describing preprocessing steps
        - beh_mask list of epochs that describes which trial types to keep (HIT / MISS / FA etc.)
    
    Return a dictionary with:
        1) baphy manager
        2) NEMS recording
        3) Spike count dictionary, collapsed over period of interest
        4) target epochs
        5) catch epochs
    """
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=bp.which_rawids(site))

    # set NEMS recording options
    options['resp'] = True
    options['pupil'] = True
    options['stim'] = False
    options['rasterfs'] = options.get('rasterfs', bp.rasterfs(batch))

    # set spike count dict options
    if dict_ops is None:
        dict_ops = {}
        dict_ops['start'] = int(bp.onset(batch) * options['rasterfs'])
        dict_ops['end'] = int(bp.offset(batch) * options['rasterfs'])

    # Load recording
    rec = manager.get_recording(recache=recache, **options)
    rec['resp'] = rec['resp'].rasterize()
    if batch==302:
        c, _ = parse_cellid({'cellid': site, 'batch': batch})
        rec['resp'] = rec['resp'].extract_channels(c)

    # Mask the data that will be extracted, to help with preprocessing?
    if beh_mask is None:
        beh_mask = bp.beh_mask(batch)
    rec = rec.and_mask(beh_mask)
    rec = rec.apply_mask(reset_epochs=True)

    # Do preprocessing on the recording (this will make interfacing with NEMS much easier down the line)
    if preprocessor is not None:
        rec = pp.wrapper(rec, preprocessor, dict_ops)

    # Extract / collapse spike counts
    targets, catch, targets_str, catch_str = bp.epoch_names(rec, manager, batch)
    d = rec['resp'].extract_epochs(targets+catch, mask=rec['mask'])
    d = {k: v[:, :, dict_ops['start']:dict_ops['end']].mean(axis=-1) for (k, v) in d.items()}

    # build masks that mirror d for extract subsets of the loaded data
    active_epochs = ['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL', 'FALSE_ALARM_TRIAL']
    passive_epochs = ['PASSIVE_EXPERIMENT']
    sigs = {}
    for ep in active_epochs+passive_epochs:
        dm = rec['resp'].epoch_to_signal(ep).extract_epochs(targets+catch, mask=rec['mask'])
        dm = {k: v[:, :, 0].astype(bool) for (k, v) in dm.items()}
        sigs[ep] = dm

    # make active mask
    active_mask = sigs['HIT_TRIAL'].copy() # place holder
    for k in active_epochs:
        for k2 in sigs[k].keys():
            active_mask[k2] = active_mask[k2] | sigs[k][k2]
    # make passive mask
    passive_mask = sigs['PASSIVE_EXPERIMENT'].copy() # place holder
    for k in passive_epochs:
        for k2 in sigs[k].keys():
            passive_mask[k2] = passive_mask[k2] | sigs[k][k2]
    
    # get BAPHY TRIAL / RT for each sound. Shape the same as the dictionary
    log.info("Loading reaction times for each target")
    baphy_trials = [e for e in rec['resp'].epochs.name if 'BAPHYTRIAL' in e]
    bt_dict = {}
    rt_dict = {}
    bev = manager.get_behavior_events(**{'rasterfs': 20})
    for e in targets+catch:
        bt_dict[e] = (np.nan * np.ones(sigs['HIT_TRIAL'][e].shape[0])).astype(str)
        rt_dict[e] = np.nan * np.ones(len(bt_dict[e]))
        me = [rec['resp'].epoch_to_signal(b).extract_epoch(e)[:,0,0] for b in baphy_trials]
        for i, m in enumerate(me):
            bt_dict[e][m] = baphy_trials[i]
        # only care about TARGET reaction times
        if 'TAR_' in e:
            for i, b in enumerate(bt_dict[e]):
                trial = int(b.strip('BAPHYTRIAL').split('_')[0])
                fidx = int(b.strip('BAPHYTRIAL').split('_')[1].strip('FILE'))-1
                if 'RT' not in bev[fidx].keys():
                    # passive file
                    pass
                else:
                    df = bev[fidx]
                    df = df[df.Trial==trial]
                    df = df[df.name.str.contains('Reference') | df.name.str.contains('Target') | df.name.str.contains('Catch')]
                    df = df[~df.name.str.contains('PreStimSilence') & ~df.name.str.contains('PostStimSilence')]
                    rt_dict[e][i] = df.iloc[-1]['RT']

    # Return results
    data = {
        'manager': manager,
        'rec': rec,
        'd': d,
        'RT': rt_dict,
        'BAPHYTRIAL': bt_dict,
        'active': active_mask,
        'passive': passive_mask,
        'TRIAL_masks': sigs,
        'targets': targets,
        'targets_str': targets_str,
        'catch': catch,
        'catch_str': catch_str
    }

    del dict_ops

    return data
    