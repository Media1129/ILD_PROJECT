# ILD_Project

## Activate virtual environment
```sh
source activate tensorflow
python resnet_train.py
python resnet_predict.py resnet50_final.h5 my_test_image.png
```
* [reference github](https://github.com/sebastianbk/finetuned-resnet50-keras)
* [reference web](https://heartbeat.fritz.ai/how-to-fine-tune-resnet-in-keras-and-use-it-in-an-ios-app-via-core-ml-ee7fd84c1b26)
    * ctrl+shift+T open new terminal
    * pip list | grep Keras


## DataPath
* data
    * train 
    * valid
* ResNet_FineTune.ipynb
## Keras flow_from_dictionary  
```sh
train_generator = train_datagen.flow_from_directory(
    directory=r"./train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
```
* **color_mode**
    * **grayscale** (either black and white or grayscale set)
    * **rgb** (three color channels)
* **batch_size**
    * No. of images to be yielded from the generator per batch
* **class_mode** 
    * **binary** (only two classes)
    * **categorical** (not only two class)
    * **input** (developing an Autoencoder system, both input and the output would probably be the same image)
* shuffle
    * Set True if you want to shuffle the order of the image that is being yielded, else set False
* seed
    * Random seed for applying random image augmentation and shuffling the order of the image
* [tutorial web page](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)
## Small data
```sh
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
```
* **rotation_range** is a value in degrees (0-180), a range within which to randomly rotate pictures
* **width_shift and height_shift** are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
* **rescale** is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor
* **shear_range** is for randomly applying shearing transformations
* **zoom_range** is for randomly zooming inside pictures
* **horizontal_flip** is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures)
* **fill_mode** is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.

* [reference web](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
