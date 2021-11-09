# Image-Captioning
Keras/Tensorflow Image Captioning application using CNN and Transformer as encoder/decoder. </br>
In particulary, the architecture consists of three models:
1. **A CNN**: used to extract the image features. In this application, it used EfficientNetB0 pre-trained on imagenet.
2. **A TransformerEncoder**: the extracted image features are then passed to a Transformer encoder that generates a new representation of the inputs.
3. **A TransformerDecoder**: it takes the encoder output and the text data sequence as inputs and tries to learn to generate the caption.
## Dataset 
The model has been trained on 2014 Train/Val COCO dataset.
You can download the dataset [here](https://cocodataset.org/#download). Note that test images are not required for this code to work.</br></br>
Original dataset has 82783 train images and 40504 validation images; for each image there is a number of captions between 1 and 6. I have preprocessing the dataset per to keep only images that have exactly 5 captions. In fact, **_the model has been trained to ensure that 5 captions are assigned for each image_**. After this filtering, the final dataset has 68363 train images and 33432 validation images.</br>
Finally, I serialized the dataset into two json files which you can find in:</br></br>
`COCO_dataset/captions_mapping_train.json` </br>
`COCO_dataset/captions_mapping_valid.json` </br></br>
Each element in the _captions_mapping_train.json_ file has such a structure :</br>
`"COCO_dataset/train2014/COCO_train2014_000000318556.jpg": ["caption1", "caption2", "caption3", "caption4", "caption5"], ...` </br></br>
In same way in the _captions_mapping_valid.json_ :</br>
`"COCO_dataset/val2014/COCO_val2014_000000203564.jpg": ["caption1", "caption2", "caption3", "caption4", "caption5"], ...` </br>
## Dependencies
I have used the following versions for code work:
* python==3.8.8
* tensorflow==2.4.1
* tensorflow-gpu==2.4.1
* numpy==1.19.1
* h5py==2.10.0
## Training
To train the model you need to follow the following steps :
1. you have to make sure that the training set images are in the folder `COCO_dataset/train2014/` and that validation set images are in `COCO_dataset/val2014/`.
2. you have to enter all the parameters necessary for the training in the `settings.py` file.
3. start the model training with `python3 training.py`

### My settings
For my training session, I have get best results with this `settings.py` file :
```python
# Desired image dimensions
IMAGE_SIZE = (299, 299)
# Max vocabulary size
MAX_VOCAB_SIZE = 2000000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512
# Number of self-attention heads
NUM_HEADS = 6
# Per-layer units in the feed-forward network
FF_DIM = 1024
# Shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 512
# Batch size
BATCH_SIZE = 64
# Numbers of training epochs
EPOCHS = 14

# Reduce Dataset
# If you want reduce number of train/valid images dataset, set 'REDUCE_DATASET=True'
# and set number of train/valid images that you want.
#### COCO dataset
# Max number train dataset images : 68363
# Max number valid dataset images : 33432
REDUCE_DATASET = False
# Number of train images -> it must be a value between [1, 68363]
NUM_TRAIN_IMG = 68363
# Number of valid images -> it must be a value between [1, 33432]
# N.B. -> IMPORTANT : the number of images of the test set is given by the difference between 33432 and NUM_VALID_IMG values.
# for instance, with NUM_VALID_IMG = 20000 -> valid set have 20000 images and test set have the last 13432 images.
NUM_VALID_IMG = 20000
# Data augumention on train set
TRAIN_SET_AUG = True
# Data augmention on valid set
VALID_SET_AUG = False
# If you want to calculate the performance on the test set.
TEST_SET = True

# Load train_data.json pathfile
train_data_json_path = "COCO_dataset/captions_mapping_train.json"
# Load valid_data.json pathfile
valid_data_json_path = "COCO_dataset/captions_mapping_valid.json"
# Load text_data.json pathfile
text_data_json_path  = "COCO_dataset/text_data.json"

# Save training files directory
SAVE_DIR = "save_train_dir/"
```
I have training model on full train set (68363 train images) and 20000 valid images but you can train the model on a smaller number of images by changing the NUM_TRAIN_IMG / NUM_VALID_IMG parameters to reduce the training time and hardware resources required.

### Data augmention
I applied data augmentation on the training set during the training to reduce the generalization error, with this transformations (this code is write in `dataset.py`) :
```python
trainAug = tf.keras.Sequential([
    	tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.05, 0.15)),
    	tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
	tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
	tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.10, 0.10))
])
```
You can customize your data augmentation by changing this code or disable data augmentation setting `TRAIN_SET_AUG = False` in `setting.py`. 
### My results

These are the results on test set (13432 images):
```
loss: 11.8024 - acc: 0.5455
```

These are good results considering that for each image given as input to the model during training, **the error and the accuracy are averaged over 5 captions**. However, I spent little time doing model selection and you can improve the results by trying better settings. </br>
For example, you could :
1. change CNN architecture.
2. change SEQ_LENGTH, EMBED_DIM, NUM_HEADS, FF_DIM, BATCH_SIZE (etc...) parameters.
3. change data augmentation transformations/parameters.
4. change optimizer and learning rate scheduler.
5. etc...

**N.B.** I have saved my best training results files in the directory `save_train_dir/`.
## Inference
After training and saving the model, you can restore it in a new session to inference captions on new images. </br>
To generate a caption from a new image, you must :
1. insert the parameters in the file `settings_inference.py`
2. run `python3 inference.py --image={image_path_file}`

## Results example
Examples of image output taken from the test set.
| a large passenger jet flying through the sky             |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/2.jpg)

| a man in a white shirt and black shorts playing tennis             |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/10.jpg)  


| a person on a snowboard in the snow             |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/15.jpg)  

| a boy on a skateboard in the street            |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/20.jpg)  

| a black bear is walking through the grass            |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/4.jpg)  


| a train is on the tracks near a station            |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/14.jpg)  
