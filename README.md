# kne_detectability

This repo contains the code used in https://arxiv.org/pdf/2004.06137.pdf

`kne_detectability_sim.py` uses simsurvey (https://github.com/ZwickyTransientFacility/simsurvey). It simulates an optimitic toO follow up for KN after starting up to 3 days from merger. It calles kilonova models from https://github.com/mbulla/kilonova_models and uses the 2 component model grid https://github.com/mbulla/kilonova_models to simulattee the kilonova lightcurve. 

`surveyplan.py` creates the observattion log for the simulation. 

