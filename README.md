# LILI

Image logs can provide critical information to reduce drilling risk, as they allow for early detection of structures such as faults or fractures. This kind of information can be critical during drilling operations, but the time required to manually interpret the data and the subjectivity of interpretations can undermine the timeliness of delivery of results. The proposed methodology is a supervised Deep Learning - based method built on U-Net architecture for segmentation of image logs acquired while drilling, i.e., automated detection of geological edges in borehole images. The proposed network has been trained on synthetic data and tested on field data. Curriculum learning involves a pre-training step on simple cases and a fine-tuning step on more complex ones, whereas standard learning is composed of one single step on a set of cases of random complexity.  The results of curriculum learning show that the network’s predictions return more populated segmentation maps than those of the network trained with standard training and can help the development of a fully automated system for the interpretation of image logs.
- # in the U-Net folder you can find the following python files:
  - data_both: python file to generate synthetic dataset during standard learning of the U-Net architecture (SL-PickNet in the paper)
  - data_complex: python file to generate complex synthetic dataset tu use in fine-tuning the U-Net (CL-PickNet in the paper) following curriculum       learning
  - data_easy:  python file to generate simple synthetic dataset tu use in pre-training the U-Net  (CL-PickNet in the paper) following curriculum       learning
  - data_test: test file to test the program on server.
  - NN_complex: CL-PickNet finetuning file
  - NN_easy: CL-PickNet pre-training file
  - NN_both: SL-Picknet training file
  - CL animation: segmentation results of CL-PickNet on windowed field dataset
  - SL animation:  segmentation results of SL-PickNet  on windowed field dataset
  - CL-PickNet fine-tuned saved model
  - SL-PickNet saved model
- #  in the 01_MC_serverp05folder you can find the following python files:
  -  01_train: file for FitNet05 training
  -  fig, animations folder containing figures of Montecarlo Dropout method results and animations of features correlation of Fitnet05 on CL and SL segmentation maps
  -  FitNet05 model: the saved model that produced the results in the paper.
