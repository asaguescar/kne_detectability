import pandas as pd
import numpy as np

mjd_start = 58239.5
field_id = 500

jd, filterid, fieldid, rcid, programid = [], [], [], [], []
for k in range(8):
    for i in range(2):
        if i==0:
            a = 0
            bandas = ['ztfg', 'ztfr', 'ztfi']
        elif i==1:
            a = 2 + np.random.uniform(1,3)
            bandas = ['ztfg', 'ztfr', 'ztfi']

        jd.extend([mjd_start + k + (l/4+a) / 24. for l in range(3)]) # Each one hour
        fieldid.extend([field_id for l in range(3)])
        filterid.extend(bandas)

limMag = 25

df = pd.DataFrame()
df['jd'] = jd
df['limMag'] = np.ones(len(jd))*limMag
df['filterid'] = filterid
df['fieldid'] = fieldid
df['comment'] = ['']*len(jd)

df.to_csv('../data/surveyplan.csv')
