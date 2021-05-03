"""
Helpers for preprocessing NEMS recordings
    * zscoring / centering data
    * regressing out state
"""
import charlieTools.preprocessing as preproc
import numpy as np

def wrapper(rec, preprocessor):
    """
    parse preprocessor string, call appropriate fns, return new recording
    """
    options = preprocessor.split('-')[1].split('.')
    for op in options:
        if op == 'pr':
            # remove pupil
            rec = regress_pupil(rec)
        elif op == 'zscore':
            # zscore data
            rec = normalize(rec, zscore=True)
        elif op == 'center':
            # center data
            rec = normalize(rec, zscore=False)
        else:
            raise ValueError(f"Preprocessing option {op} not defined")

    return rec


def regress_pupil(rec):
    return preproc.regress_state(rec, state_sigs=['pupil'])


def normalize(rec, zscore=False):
    alldata = rec['resp'].extract_epochs(['TARGET', 'REFERENCE', 'CATCH'], mask=rec['mask'], allow_incomplete=True)
    alldata = np.concatenate([v for k, v in alldata.items()], axis=0)
    m = np.nanmean(alldata, axis=(0, 2, 3))
    sd = np.nanstd(alldata, axis=(0, 2, 3))
    sd[sd==0] = 1

    if zscore:
        data = rec['resp']._data
        data = data.T - m 
        data = data / sd
        rec['resp'] = rec['resp']._modified_copy(data.T)
    
    else:
        data = rec['resp']._data
        data = data.T - m 
        rec['resp'] = rec['resp']._modified_copy(data.T)
    
    return rec


