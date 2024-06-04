# PAPN: Pyramid Attention Enhanced Prototype Network for Few-Shot Time Series Classification for Manufacturing Process
This is a repository containing code for paper PAPN: Pyramid Attention Enhanced Prototype Network for Few-Shot Time Series Classification for Manufacturing Process.
The code is developed based on the learn2learn framework.

The timeproto.py can train, validate and test the datasets from the UCR time series archive using prototype network.
The feature extractor can be chosen from store.py.

The CWRU.py uses prototype network to train, validate and test the refined CWRU dataset (Statistical features).
The feature extractor can be chosen from store.py.

MAML.py and Reptile.py can be used to train, validate and test the MAML and Reptile model on time series data.
The model can be chosen from store_1.py.

All the dataset used in the experiment can be found in Data.zip.
