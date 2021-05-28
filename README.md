# Neural-Image-Descriptor
Image captioning is the process by which we can get a textual description of an image.  

Computer vision has become universal in our society, with applications in various fields. In this project, we focus on one of the visual recognition facets of computer vision, i.e. image captioning. The problem of generating language descriptions for visual data has been studied from a long time but in the field of videos. In the recent few years emphasis has been lead on still image description with natural text. Due to the recent growth in the field of object recognition, the errand of scene depiction in a picture has turned out to be simpler.   

The point of the venture was to train convolutional neural systems with a few several
hyperparameters and apply it on a tremendous dataset of pictures (Image-Net), and join the
consequences of this picture classifier with an intermittent neural system to produce an
inscription for the grouped picture. In this report we present the definite engineering of the
model utilized by us. We accomplished a BLEU score of 56 on the Flickr8k dataset while the
cutting-edge results lay at 66 on the dataset.  

model_path = `./models`  
feature_path = `./data/feats.npy`  
captions_path = `./data/results.token`  
vgg_path = `./data/vgg16.tfmodel`  

### Dataset
Dataset on which model trained is `Flickr30k`, Website : `http://bryanplummer.com/Flickr30kEntities/`  

Clone the Flickr30k Dataset from `https://github.com/BryanPlummer/flickr30k_entities.git`  

# Image Classification
This was a sub project done for understanding the classification of Images in different classes.  
This sub-project is done on `Fashion MNIST` dataset.  
Download Dataset from `https://www.kaggle.com/zalando-research/fashionmnist`.  

Learn More about this Image Classification from :- `https://medium.com/swlh/exploring-fashion-mnist-with-tensorflow-and-keras-aa780b766149`

## References -  
1. https://medium.com/deep-math-machine-learning-ai/chapter-8-0-convolutional-neural-networks-for-deep-learning-364971e34ab2
