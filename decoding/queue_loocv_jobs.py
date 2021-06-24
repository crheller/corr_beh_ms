import nems.db as nd
import numpy as np

from settings import B302_sites, B307_sites, B324_sites

sites = [B302_sites, B307_sites, B324_sites]
batches = [302, 307, 324]

force_rerun = True

modellist = [
    'leaveOneOut_pp-zscore_dDR2-allTargets-noise.target-mask.h+c+m+i+p_wopt-mask.h+c+m+i+p',
]

script = '/auto/users/hellerc/code/projects/corr_beh_ms/decoding/leave_one_out_cache.py'
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'

for batch, site in zip(batches, sites):
    nd.enqueue_models(celllist=site,
                    batch=batch,
                    modellist=modellist,
                    executable_path=python_path,
                    script_path=script,
                    user='hellerc',
                    force_rerun=force_rerun,
                    reserve_gb=1)