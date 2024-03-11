# Compost, Recycle, Trash Categorizer (Tentative Name)
## Milestone 4: Model 2 and Evaluation
<a target="_blank" href="https://colab.research.google.com/drive/11Fny_6xKCY2_ddIDxwC6iGdPsJowVTVW?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Model Evaluation
There were no changes made to the data, labels, and loss function as they were sufficient for our 2nd model.
The 2nd model had an accuracy of about 62% for testing accuracy and a 65% accuracy in predicting the training data. For the validation sample, it had an accuracy of about 66% in predicting the validation targets. Given that the accuracy of the testing dataset is slightly less than the accuracy of the training and validation dataset, this suggests that the model slightly overfits. 
Compared to our first model, the testing accuracy is slightly lower than the 1st model's suggesting that the first model was slighly better at predicting the trash type. In the first model, it was found that the data didn't overfit while the second does sligthly. 
As for error, we looked at the log loss which inicated that the test loss was larger than the training and validation loss suggesting that there is a slight overfitting going on in the model.
### Results of Hyper Parameter Tuning
We did hyperparameter tuning to find the optimal parameters for the model. Through this, we were able to get better parameters for our deep neural network to better predict the trash type. This allowed us to know the best number of units and activation functions for each layer. It also let us see the best learning rate and loss function. 
### Next Model
For the next model, we are thinking of 
### Conclusion
Overall, for image data, a deep neural network isn't the best type of model to clasify as seen by how the testing accuracy was higher in our first CCN model as opposed to our second model which is a deep neural network. This suggests that adding some convolution layers could be helpfull in better predicting the trash type. 

## Milestone 3: Preprocessing 
<a target="_blank" href="https://colab.research.google.com/drive/1KEcvWCYkV52NPyeS-_5xmDSmxJIV3ke8?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Model evaluation
For our model, we decided to preprocess the pictures by changing the image sizes to be 180 x 180 and by converting the pictures into greyscaled images. After preprocessing the images, we trained our model with the new data and achieved an accuracy score of 0.63. Although this score is better than the score attained by randomly guessing, we hope to improve our model by having the images colored instead. We believe that using greyscale for the images could have made some features harder to detect, possibly leading to our lower accuracy score. Since both the training and testing accuracies are lower, being around 0.6, the model does underfit. 
### 2 New Models
- For one model, we will try having separate channels so that the colors can be considered in the model since greyscaling may have caused the accuracy to go down
- For the second, we will try changing the activation functions of the layers to see which one is optimal in minimizing the loss. Additionally, altering the units of the layer to better categorize the trash. This is to tweak the model to be better fitting

## Milestone 2: Data Expoloration
<a target="_blank" href="https://colab.research.google.com/drive/1ppVIFZKg99gVINq3GeW3XbGLlXQuLOnY?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

 ### Preprocessing
 Some steps we have taken and will take in the future is checking whether the image sizes are uniform. They are uniform in this data set so making them uniform is not needed. We also check for outliers such as much larger or smaller images or black and white images. None of these were found. All images are uniformally sized and in color. In our data there is a category of miscellaneous trash. For this, potential preprocessing steps could be to use data transformation, normalization, and stadardization methods and functions to better train our model to improve its accuracy. Additionally, the data has 9 categories right now, so these categories need to be grouped together depending on whether they are recyclable, compostable, or landfill. For this task we have this source to help decide how to group these categories:
Trash, recycle, compost guide: https://www.sandiego.gov/sites/default/files/cowr-guide-english.pdf

# RealWaste Dataset

RealWaste is an image dataset assembled from waste material received at the Whyte's Gully Waste and Resource Recovery facility in Wollongong NSW Australia.

If you use our dataset in your work, please cite our original work: [RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning](https://www.mdpi.com/2078-2489/14/12/633).

Please note, this dataset is licenced under CC BY-NC-SA 4.0.

The dataset is composed of the following labels and image counts:
  - Cardboard: 461
  - Food Organics: 411
  - Glass: 420
  - Metal: 790
  - Miscellaneous Trash: 495
  - Paper: 500
  - Plastic: 921
  - Textile Trash: 318
  - Vegetation: 436

The above labelling may be further subdivided as required, i.e., Transparent Plastic, Opaque Plastic.
Trash, recycle, compost guide: https://www.sandiego.gov/sites/default/files/cowr-guide-english.pdf

