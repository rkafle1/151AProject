# Compost, Recycle, Trash Categorizer
## Introduction
Identifying whether trash can be recycled or composted can be a hassle as the state and material of the trash dictates what you can do with it. The goal of our project is to reduce this stress, and be able to classify an image of trash as recyclable, compostable, or landfill. By creating this model, clear images of trash can be classified to help us ensure we are disposing of waste properly and improving the environment. We chose this project due to its potential environmental impact by facilitating proper waste management practices. Having an accurate predictive model can assist in automating the sorting process, leading to more efficient recycling and waste distinction efforts. Additionally, the model can contribute to reducing contamination in recycling systems and promote sustainable practice.


## Methods
(this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, Model 3.
(Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, Model 3)
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods

### Data Exploration
<a target="_blank" href="https://colab.research.google.com/drive/1ppVIFZKg99gVINq3GeW3XbGLlXQuLOnY?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The RealWaste dataset comprises images of waste items collected from the Whyte's Gully Waste and Resource Recovery facility in Wollongong NSW, Australia.

The main goal was seeing what images we had and what their layout/size was. We first began by plotting the image sizes. Then we looked into the color distribution of different categories of trash the data set provided. Additionally, we looked into the number of images per each trash type. 

We also looked into whether their were any outliers in image size and label consistency. 

Our final step was displaying some randomly chosen images from each trash category and displaying them.

### Preprocessing
We started by grouping the trash categories we were given via the dataset and grouping them as recycle, compost, or landfill. This grouped the images into 3 groups. 

To group these categories: Trash, recycle, compost guide: https://www.sandiego.gov/sites/default/files/cowr-guide-english.pdf 


The preprocessing implementation is shown before every colab moving forward. To refer to the code, look at the model 1 - 3 notebooks.


Other preprocessing we implemented was gray scaling and sizing down images before passing them into the model.


### Model 1 Convolutional Neural Network on Grayscale
...

### Model 2 Convolutional Neural Network on RGB
...

### Model 3 K-Nearest Neighbors
<a target="_blank" href="https://colab.research.google.com/drive/1D1gvBfgVLAjMIjJTO7Bi0nL5cj_D9U8x?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Our K-nearest neighbors model examines the RGB distances for each pixel. This model allows us to group images that have similar features and classify them as one of the three trash types. For the model, we preprocess image input into a flattened one-dimensional vector containing each original pixel's respective RGB values, so the dataframe contains observations of our images in which the features are these pixel values. The images will be classified based on the composition of RGB values, with images of similar colors being more related. Our chosen parameter was k=11 neighbors as observed from multiple trials with different k values. In order to find a good k value, we ran multiple trials on the testing set to find the best accuracy of the first 30 k values.
#### Trials on testing set:

![image](https://github.com/rkafle1/151AProject/assets/88344031/b5399723-94a3-432b-b4cd-2550f5e3794b)

## Results
This will include the results from the methods listed above (C). You will have figures here about your results as well.
No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

### Model 1 Convolutional Neural Network on Grayscale
...

### Model 2 Convolutional Neural Network on RGB
...

### Model 3 K-Nearest Neighbors
Our model performed with an accuracy of 70% on the testing set and 69% on the validation set with k=11 neighbors.
#### Testing results:

![image](https://github.com/rkafle1/151AProject/assets/88344031/6562151a-a943-4f89-bd37-84d0e637d572)
#### Validation results:

![image](https://github.com/rkafle1/151AProject/assets/88344031/24242705-9bcf-4871-9373-637c848f30c7)


## Discussion
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

## Conclusion
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

## Collaboration
This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!
Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.

<hr><hr><hr>

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
For the next model, we are thinking of using a k-nearest neighbors model that examines the rgb distances for each pixel. We chose this model since it allows us to group images that have similar features and classify them as one of the three trash types. We intend to approach the model by processing image input into a flattened one-dimensional vector in which every 3 values in the vector represent the respective rgb values for each pixel in the original message. With this model, we anticipate that the images will be classified based on the composition of rgb values, with images of similar colors being more related.
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

