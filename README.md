# Wearable Stress Measurement System Application

Last updated: 18 May 2021

Application for the wearable stress measurement system for the 2020/2021 EBME
senior design project of Team 4 at Case Western Reserve University. The
deep-learning model is an implementation of the deep
reconstruction-classification network [1]. This model was selected because it
satisfies all user needs for a wearable stress measurement system designed for
the ICU.

The following README provides a high level overview of the source code
organization and what tasks remain after the first iteration of design. Please
contact Ryan Tatton at rdt17@protonmail.com with any questions or inquiries.

## Organization

All source code is contained within the `wsma` directory.

The `app` package contains `app.py`, which is the user interface application of
the system. The `connect.py` module containers the data access
class `WearableDevice`, which allows data transduced and processed by the
wearable device to be received by the application device.

The `model` package contains the `drcn.py` module, which includes the `DRCN`
class, and is the model used by the wearable stress measurement system to infer
patient stress. Note that the multi-task learning convolutional neural network
described in [1] is not currently implemented. The `DRCN` class, while not
formally verified with unit testing, is completed and should not require
modification, beyond (perhaps) refactoring to expose several parameters of the
individual networks layers. Many of the parameter values hard-coded in the
constructors are specified by [1].

The `preprocess.py` module contains functions used for preprocessing the
datasets. Note that while [1] provides several source-target comparisons in
their experiments, this library assumes the use of the Distracted Driving
dataset (source) and the MIT Driver Stress dataset (target) specified in the
paper since it scored the highest. The preprocessing of the MIT dataset is
completed, but the Distracted Driving dataset is not. The complete dataset
contains many other files, which are not necessary for this model. Be aware that
the full dataset is about 2 TB, but it appears that much of that is from the
irrelevant video files. It is suspected that once the irrelevant data files are
removed, the dataset will become significantly more manageable, and may even be
small enough to keep in memory while training, like the MIT dataset.

The `train.ipynb` references some online tutorials to help optimize training
times with Google Colab. Please refer to those tutorials. This notebook is not
complete. Once the preprocessing code has been implemented, the rest of the
notebook can be written, which will simply entail making calls to `drcn. py`
, `preprocess.py`, and `train.py`. Google Colab is highly encouraged for
training since it offers GPU computing for free and is fairly simple learn.

## Remaining Tasks

In summary, the following are what remain to be done:

- Implement preprocessing for the Distracted Driving dataset
- Finish implementing the `train.py` functions
- Complete the Google Colab notebook
- Train the model

## References

[1] A. Saeed, T. Ozcelebi, J. Lukkien, J. B. F. van Erp and S. Trajanovski,
"Model Adaptation and Personalization for Physiological Stress Detection," 2018
IEEE 5th International Conference on Data Science and Advanced Analytics (DSAA),
2018, pp. 209-216, doi: 10.1109/DSAA.2018.00031.