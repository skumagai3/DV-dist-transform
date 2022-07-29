# DV-dist-transform
Repository containing minimal working example of the 2-D test U-Net that takes in a density field and spits out a distance transform field.


Model trained with [0,1] scaling of log10 density and distance transform:
![2DTEST-preproc-pred-comp](https://user-images.githubusercontent.com/38794996/181806535-85e87147-bfb4-44e8-8a1b-b10aaad40064.png)


Model trained with [0,1] scaling of density and distance transform:
![2DTEST-nolog-preproc-pred-comp](https://user-images.githubusercontent.com/38794996/181814225-29d9c449-2840-4b12-9d7e-2028646f2810.png)


DV-2D.ipynb notebook loads in 256 images (density slices) from Illustris TNG and the corresponding distance transform fields. It runs training with a model (architecture in build_model at the end of nets.py) with mean squared error loss and a linear activation function. It then runs predictions on the data it trained on and produces a comparison figure.

Ongoing and Future work: 
- Determining what the best type of preprocessing is. At the moment, the density and distance transform fields are minmaxed to a range of [0,1], but taking the log of the density before that step may or may not help.
- Is MSE the best loss function to use? Easy alternatives built into Keras include MAE, MAPE, etc. (see [Keras reg losses](https://keras.io/api/losses/regression_losses/))
- Custom loss function(s) for bespoke networks? By bespoke I mean purpose-built, like a model that would reward voxels at the center of voids more than those at the edges or vice versa, etc. (see [losses](https://keras.io/api/losses/)) 
- Does dropout help with generalizability? It has been proven to for fully connected networks (but not necessarily for CNNs, see [this article](https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2)). Right now there are dropout layers after each convolution or deconvolution block, but these can be easily deactivated.
- What is the receptive field at the bottleneck? Or in other words (I think), how much of the input image or volume corresponds to one pixel at the bottleneck (lowest layer)? [reference](https://www.baeldung.com/cs/cnn-receptive-field-size)
- Why isn't this translating into 3D? 3D U-Net for this same task is not converging in training and the predictions are all one value except for one row at the bottom of the volume. Is this a reconstruction difficulty or some other issue? Have we let the 3D network train long enough? 
- Should we be seeking to minimize validation loss or just loss? Odd behavior during training:
![2DTEST-preprocaccloss_vs_epochs](https://user-images.githubusercontent.com/38794996/181814564-b2eea425-f877-4f8b-971a-3c391cf4f716.jpg) Note that the current DeepVoid and 2DTest models all save the model with the best validation (test set specified in the model.fit call) loss.

## CONTACT ME
If you have problems running the notebook, or other questions about training/predicting/the DeepVoid project in general, message me on the DeepVoid collaboration Slack or by email at sk3993@drexel.edu.
