In thid folder you find the data to reproduce the results in the paper.
These are .npy files that can be loaded in a python file using numpy.

- LWD data interval: for confidentiality agreement with Eni, I can't share the whole dataset available. I share the 100 samples interval plotted in the paper figures
- restored segmentation maps: the segmentation maps from SL- and CL- PickNets for the 100 samples interval (see above)
- FitNet05 on segmentation maps:
    - FitNet05 x SL-PickNet: features correlation results on SL segmentation map
    - FitNet05 x CL-PickNet: features correlation results on CL segmentation map
    These data have shape (101, 4, 16), where:
        - 101 is the number of windows in which we perform features correlation in the 100 samples interval
        - 4 is the number of elements returned from FitNet05 in each window:
            - 0: predicted curve#1
            - 1: standard deviation for each points on curve#1
            - 2: predicted curve#2
            - 3: standard deviation for each points on curve#2
        - 16 is the number of points in each curve
 