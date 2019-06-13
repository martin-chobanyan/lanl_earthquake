# lanl_earthquake
My approach to the [Kaggle LANL Earthquake competition](https://www.kaggle.com/c/LANL-Earthquake-Prediction)

The competition involves predicting when the next seismic activity (earthquake) will occur in a laboratory experiment.
The training data is a large sequence of acoustic values at a high resolution. Each acoustic value in the sequence is 
labeled with the number of seconds until the next earthquake. The goal is to train a model to accurately detect when the next earthquake occurs,
given very small segments of acoustic values from independent test runs.

My approach implemented in this repo is a deep learning model composed of three parts:
- gru on features extracted from spike segments
- acoustic 1d conv
- spectogram 2d conv

For more details see the python notebook.
