import nems.db as nd
import numpy as np

from settings import ALL_SITES

force_rerun = True

modellist = [
    'leaveOneOut_mask.h+m+p_pp-zscore-pr_dDR4-allTargets-noise.target-mask.p+c_wopt-mask.p',

]

script = '/auto/users/hellerc/code/projects/corr_beh_ms/decoding/leave_one_out_cache.py'
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modellist,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)