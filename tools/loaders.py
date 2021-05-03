'''
Load behavior data. Options depend on the batch, since each task is slightly different.
'''
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import parse_cellid

import tools.batch_params as bp
import tools.preprocessors as pp

def load_data(site, batch, preprocessor=None, beh_mask=None, dict_ops={}, recache=False, **options):
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
    options['rasterfs'] = options.get('rasterfs', bp.rasterfs(batch))

    # set spike count dict options
    dict_ops['start'] = int(dict_ops.get('start', bp.onset(batch)) * options['rasterfs'])
    dict_ops['end'] = int(dict_ops.get('end', bp.offset(batch)) * options['rasterfs'])

    # Load recording
    rec = manager.get_recording(recache=recache, **options)
    rec['resp'] = rec['resp'].rasterize()
    c, _ = parse_cellid({'cellid': site, 'batch': batch})
    rec['resp'] = rec['resp'].extract_channels(c)

    # Mask the data that will be extracted, to help with preprocessing?
    if beh_mask is None:
        beh_mask = bp.beh_mask(batch)
    rec = rec.and_mask(beh_mask)
    rec = rec.apply_mask(reset_epochs=True)

    # Do preprocessing on the recording (this will make interfacing with NEMS much easier down the line)
    import pdb; pdb.set_trace()
    if preprocessor is not None:
        rec = pp.wrapper(rec, preprocessor)

    # Extract / collapse spike counts
    targets, catch, targets_str, catch_str = bp.epoch_names(rec, manager, batch)
    d = rec['resp'].extract_epochs(targets+catch, mask=rec['mask'])
    d = {k: v[:, :, dict_ops['start']:dict_ops['end']].mean(axis=-1) for (k, v) in d.items()}

    # Return results
    data = {
        'manager': manager,
        'rec': rec,
        'd': d,
        'targets': targets,
        'targets_str': targets_str,
        'catch': catch,
        'catch_str': catch_str
    }

    return data
    