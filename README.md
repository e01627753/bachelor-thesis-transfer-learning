# Bachelor Thesis: Transfer Learning

  ### 1. Research Question:
  How strong is the impact of transfer learning regarding the efficiency and effectiveness of convolutional neural networks, based on the example of detecting vehicle registration plates, assuming working with limited datasets?
  
  ### 2. Approach:
  The data is downloaded from the Open Images V6 database. To simulate a real environment usecase the models are trained to detect vehicle registration plates. Due to GDPR issues edge devices used for e.g. smart traffic will not be fit with real data. Thus the trainings data creation process will cost an enormous amount of work and most likely result in limited datasets. For a more realistic comparison the data will be preprocessed using an image augmentation algorithm. This step is common use for limited datasets and increases the accuracy of both models as well as reduces the likelihood of overfitting.\
  \
  To be able to compare transfer learning with training convolutional neural networks from scratch, both methods have been implemented. For transfer learning the tensorflow lite model maker library has been used. This library is optimized for edge devices and therefore suits best to this thesis.\
  The second CNN is based on the tensorflow keras library. Several different CNN structures have been tested. You can take a look at the results in the bachelor-thesis.pdf.\
  \
  The resulting models have been quantized to be compatible with edge devices and increase the inference speed. The models have been attached to the project and can be downloaded from repository/models. There are 2 versions of the 8 bit models: the quantized tflite models, which have been exported from the jupyter notebook and the TPU models. The second model is the result of recompiling the first tensorflow lite model with an Edge TPU compiler to be able to execute the models on Coral Edge TPUs.\
  \
  At the end of the bachelor thesis both methods have been analyzed in terms of efficiency and effectiveness. In this context comparing efficiency means measuring inference speed plus power consumption on 2 different edge devices (Raspberry Pi 4 and Coral Edge TPU) and comparing the storage space needed for the quantized models. All measures are set in relation to their accuracy. More complex models are more accurate but will also cost more capacity in regards of energy, space and inference time. To be able to compare different models for edge computing we need to take this trade off into account.\
  For taking the measure and running inference of the models on the edge devices a seperate Python script (execute_models.py) has been implemented.\
  The results have been summarized in the thesis itself (bachelor-thesis.pdf) but you can also take a look on the diagrams of the measures in results/.
