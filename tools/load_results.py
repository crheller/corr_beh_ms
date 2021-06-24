import pandas as pd
import os

def load_loocv(modelname, sites):
    path = '/auto/users/hellerc/results/corr_beh_ms/loocv/'
    results = []
    for site in sites:
        fn = os.path.join(path, site, modelname+'.pickle') 
        r = pd.read_pickle(fn)
        r['site'] = site
        results.append(r)
    
    return pd.concat(results)
