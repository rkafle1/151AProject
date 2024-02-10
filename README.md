# Compost, Recycle, Trash Categorizer (Tentative Name)
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

