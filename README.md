# Compost, Recycle, Trash Categorizer (Tentative Name)

...

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



# Preprocessing
- checking for outliers(any larger images-> scale it down to have uniform sizes, any black and white images)
- In our data, there are pictures of trash labeled as miscellaneous. Potential preprocessing steps could be to use data transformation, normalization and standardization methods and functions to bettern train our model and improve its accuracy. 
- recategorizing the data as recyclable, compostable and landfill
- 
