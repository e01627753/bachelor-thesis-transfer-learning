# Bachelor Thesis: Transfer Learning

  ### 1. Research Question:
  How strong is the impact of transfer learning regarding
  the efficiency and effectiveness of convolutional neural net-
  works, based on the example of detecting vehicle registration
  plates, assuming working with limited datasets?
  
  ### 2. Approach:
  To be able to compare transfer learning with training con-
  volutional neural networks from scratch, both methods will
  be implemented. For transfer learning the tensorflow lite
  model maker library will be used. Given the complexity of
  the resulting CNN (number of parameters, batch size,...) the
  second CNN will be adapted accordingly to reach the highest
  comparability possible.
  The data fitted into the models will simulate the issues we
  face in a lot of edge computing areas namingly having small
  datasets or not being able to create realistic data because of
  GDPR. In particular, the CNNs will learn the detection of
  vehicle registration plates with a limited amount of data.
  For a more realistic comparison the data will be prepro-
  cessed using an image augmentation algorithm. This step is
  common use for limited datasets and increases the accuracy
  of both models.
  At the end of the bachelor thesis both methods will be ana-
  lyzed in terms of efficiency and effectiveness. In this context
  comparing efficiency means measuring computation time
  of the learning and decision making process. Effectiveness,
  on the other hand, will be illustrated by evaluating accuracy,
  overfitting and ROC curves.
