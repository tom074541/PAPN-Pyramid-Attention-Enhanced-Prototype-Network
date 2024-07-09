The timeproto.py can train, validate and test the datasets from the UCR time series archive using prototype network.
The feature extractor can be chosen from store.py.

The CWRU.py uses prototype network to train, validate and test the refined CWRU dataset (Statistical features).
The feature extractor can be chosen from store.py.

MAML.py and Reptile.py can be used to train, validate and test the MAML and Reptile model on time series data.
The inner model can be chosen from store_1.py.

Multiple.py could run the prototype network multiple times and derive the results' mean and standard division values.

Assign feature extractor for prototype network and inner model for MAML/Reptile. Assign values for train/test shots, ways
in the main function.