# ILD_Project

## Run the Project
```sh
source activate tensorflow
python resnet_train.py
python resnet_predict.py resnet50_final.h5 NORMAL 11(folder) 58(picture num)  > outputtest
```
## Resnet keras Reference
* [reference github](https://github.com/sebastianbk/finetuned-resnet50-keras)
## DATA Description 
* Lin Doctor (ILD ' Normal lung CT Folder)
    * ILD
        * 067(1) 36
        * 120(2) 7
        * 163(3) 91
        * 192(4) 34
        * 195(5) 97
    * Normal
        * 006(11) 58
        * 011(12) 39
        * 016(13) 92
* Wu Doctor(test folder)
    * ILD
        * 196(6) 96
        * 023(7) 99
        * 076(8) 116
        * 034(9) 54
        * 197(10) 105
    * Normal
        * Normal_01(14) 130
        * Normal_02(15) 117
* ILD(735)
* Normal(436)
## code/data file
* data
    * train 
    * valid
* resnet_train.py 
* resnet_predict.py
## first train
* train 
    * 1,2,3,4,6,7,8,9(533) : 11,12,14(227)
* valid
    * 5,10(203) : 13,15(209)
## second train
* train
    * 1,4,5,8,9 (337) : 13,14,15(339)
* valid
    * 2,3(98) : 11,12(97)
* test
    * 6,7,10
* ![](https://i.imgur.com/W16QVlh.png)
* train(0.99),valid(0.92),test(not good)
## third train
* train
    *  
* valid
    *  
* test
    * 
## validation loss 
* scale the batch size
    * 16-32-64
* learning rate 
    * 0.01-0.0001 
* network 
    * two layer dense
* Add Dropout
* optimizer 
    * SGD and Adam
* val-loss: 0.62
* val-accuracy: 0.73
## Helpful Skill
* ctrl+shift+T open new terminal
* pip list | grep Keras
* create .gitignore file
    * write down the filename or foldername
* python 
    * str(integer)
    * int('str')
## Image processing 
 ```sh
img = image.load_img(img_path, target_size=(224, 224))
```
* numpy_array_dimension
    * ndarray.shape
    * eg: a.shape
* 
## What is a Batch?
* Batch Gradient Descent
    * Batch Size = Size of Training Set
* Stochastic Gradient Descent
    *  Batch Size = 1
* Mini-Batch Gradient Descent
    * 1 < Batch Size < Size of Training Set
## What Is an Epoch?
* The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset
* You can think of a for-loop over the number of epochs where each loop proceeds over the training dataset. Within this for-loop is another nested for-loop that iterates over each batch of samples, where one batch has the specified “batch size” number of samples
*  You may see examples of the number of epochs in the literature and in tutorials set to 10, 100, 500, 1000, and larger
## What's is the difference between train, validation and test set, in neural networks?
* The training and validation sets are used during training.
```sh
for each epoch
    for each training data instance
        propagate error through the network
        adjust the weights
        calculate the accuracy over training data
    for each validation data instance
        calculate the accuracy over the validation data
    if the threshold validation accuracy is met
        exit training
    else
        continue training

```
* Once you're finished training, then you run against your testing set and verify that the accuracy is sufficient
* **Training Set:** this data set is used to adjust the weights on the neural network
* **Validation Set:** 
    * this data set **is used to minimize overfitting** 
    * You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). 
    * If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training
* **Testing Set:** this data set is used only for testing the final solution in order to confirm the actual predictive power of the network.
* [overflow reference web](https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo)
## Keras Generator using flow_from_dictionary  
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
## Keras Data augmentation
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



