# ILD_Project

## Run the Project
```sh
source activate tensorflow
python resnet_train.py
python resnet_predict.py model_output/001_train.h5
```
## Resnet keras Reference
* [reference github](https://github.com/sebastianbk/finetuned-resnet50-keras)
## DATA Description 
* ![](https://i.imgur.com/Rztfsgd.png)

* Lin Doctor (ILD ' Normal lung CT Folder)
    * ILD
        * 067(1),120(2),163(3),192(4),195(5)
    * Normal
        * 006(11),011(12),016(13)
* Wu Doctor(test folder)
    * ILD
        * 196(6),023(7),076(8),034(9),197(10)
    * Normal
        * Normal_01(14),Normal_02(15)
* ILD(735),Normal(436)
## first train
* train 
    * 1,2,3,4,6,7,8,9(533) : 11,12,14(227)
* valid
    * 5,10(203) : 13,15(209)
* validation(0.7) not work
* the data i put  to train  is not balanced
## second train
* train
    * 1,4,5,8,9 (337) : 13,14,15(339)
* valid
    * 2,3(98) : 11,12(97)
* test
    * 6,7,10
* ![](https://i.imgur.com/W16QVlh.png)
* ![](https://i.imgur.com/U57u7eB.png)
    * 001_train.h5
    * batch:16
    * network layer: 32-Dropout-16-Dropout
    * lr: 0.0001
    * epoch: 150
* ![](https://i.imgur.com/jXFZCdy.png)
* ![](https://i.imgur.com/BVqLCKD.png)
    * 002_train.h5
    * batch:16
    * network layer: 128-Dropout-32-Dropout
    * lr: 0.0001
    * epoch: 200
* ![](https://i.imgur.com/E4lIR3k.png)
* ![](https://i.imgur.com/5WZIe2H.png)
    * 003_train.h5
    * batch:16
    * network layer: 128-Dropout-32-Dropout
    * lr: 0.0001
    * epoch: 1000
* ![](https://i.imgur.com/zc78PbR.png)
* ![](https://i.imgur.com/UpFou60.png)
    * 004_train.h5
    * batch:32
    * network layer: 128-Dropout-32-Dropout
    * lr: 0.0001
    * epoch: 500



* train(0.99),valid(0.92)
## third train
* train
    * 2,3,6,7(293) : 11,14,15(305)
* valid
    * 5(97) : 13(92)
* test
    * 1,4,8,9,10 : 12(39)
* ![](https://i.imgur.com/hEx9eFe.png)
* ![](https://i.imgur.com/KWEinuS.png)
    * 005_train.h5
    * batch:16
    * network layer: 32-Dropout-16-Dropout
    * lr: 0.0001
    * epoch: 150
* ![](https://i.imgur.com/NV8rTaI.png)
* ![](https://i.imgur.com/s650Rke.png)
    * 006_train.h5
    * batch:16
    * network layer: 32-Dropout-16-Dropout
    * lr: 0.0001
    * epoch: 500
* ![](https://i.imgur.com/LjfuW9X.png)
* ![](https://i.imgur.com/48pc8AX.png)
    * 007_train.h5
    * batch:16
    * network layer: 128-Dropout-32-Dropout
    * lr: 0.0001
    * epoch: 500
* ![](https://i.imgur.com/9EFpNXl.png)
* ![](https://i.imgur.com/aWx0qvs.png)
    * 008_train.h5
    * batch:16
    * network layer: 128-Dropout-32-Dropout
    * lr: 0.0001
    * epoch: 1000
* ![](https://i.imgur.com/fLY5EsZ.png)
* ![](https://i.imgur.com/7E03cKk.png)
    * 009_train.h5
    * batch:16
    * network layer: 512-Dropout-128-Dropout
    * lr: 0.0001
    * epoch: 1000
* ![](https://i.imgur.com/angzFiu.png)
* ![](https://i.imgur.com/bm6y1mC.png)
    * 010_train.h5
    * batch:16
    * network layer: 512-128
    * lr: 0.0001
    * epoch: 300
* ![](https://i.imgur.com/eixuOfQ.png)
* ![](https://i.imgur.com/U0nK9ea.png)
    * 011_train.h5
    * batch:16
    * network layer: 16-Dropout-4-Dropout
    * lr: 0.0001
    * epoch: 300
* ![](https://i.imgur.com/leNot9E.png)
* ![](https://i.imgur.com/TqReAyr.png)
    * 012_train.h5
    * batch:16
    * network layer: 16
    * lr: 0.0001
    * epoch: 300
    * relu

* ![](https://i.imgur.com/xhYhC3U.png)
* ![](https://i.imgur.com/95yD9Eb.png)
    * 013_train.h5
    * batch:16
    * network layer: 16
    * lr: 0.0001
    * epoch: 100
    * sigmoid
## four train 
* train
    * 3,5,10(293) : 11,13,14(280)
* valid
    * 1,9(90) : 15(117)
* test
    * 2,4,6,7,8 : 12(39)
* ![](https://i.imgur.com/hKgBibh.png)
* ![](https://i.imgur.com/rCxr8vF.png)
    * 014_train.h5
    * batch:16
    * network layer: 16-Dropout
    * lr: 0.0001
    * epoch: 150
    * relu
* ![](https://i.imgur.com/whwz4xi.png)
* ![](https://i.imgur.com/2bdA29P.png)
    * 015_train.h5
    * batch:16
    * network layer: 4-Dropout
    * lr: 0.0001
    * epoch: 150
    * sigmoid
* ![](https://i.imgur.com/7aBF9xn.png)
* ![](https://i.imgur.com/Hf0c8Nx.png)
    * 016_train.h5
    * batch:16
    * network layer: 32-Dropout-16-Dropout
    * lr: 0.0001
    * epoch: 150
    * relu
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
* F2: rename the file name
* Ubnutu: setting-set the automate lock screen
* **python** 
    * **str(integer) , int('str')**
    * **print("NORMAL_NUM=%d" % (NORMAL_NUM) )**
    * **for FOLDER in range(1,16):**
    * **labels = [ str(i) for i in range(1,16)]**
    * **ILD_TYPE = 'ILD' if FOLDER <= 10 else 'NORMAL'**
    * **python main.py sys.argv[1] sys.argv[2]**
## Image processing 
 ```sh
img = image.load_img(img_path, target_size=(224, 224))
```
* numpy_array_dimension
    * ndarray.shape
    * eg: a.shape
### What is a Batch?
* Batch Gradient Descent
    * Batch Size = Size of Training Set
* Stochastic Gradient Descent
    *  Batch Size = 1
* Mini-Batch Gradient Descent
    * 1 < Batch Size < Size of Training Set
### What Is an Epoch?
* The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset
* You can think of a for-loop over the number of epochs where each loop proceeds over the training dataset. Within this for-loop is another nested for-loop that iterates over each batch of samples, where one batch has the specified “batch size” number of samples
*  You may see examples of the number of epochs in the literature and in tutorials set to 10, 100, 500, 1000, and larger
### What's is the difference between train, validation and test set, in neural networks?
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
### Keras Generator using flow_from_dictionary  
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
* **class_mode**   git config --global push.default matching

To squelch this message and adopt the new behavior now, use:

  git config --global push.default simple

When push.default is set to 'matching', git will push local branches
to the remote branches that already exist with the same name.

Since Git 2.0, Git defaults to the more conservative 'simple'
behavior, which only pushes the current branch to the corresponding
remote branch that 'git pull' uses to update the current branch.

See 'git help config' and search for 'push.default' for further information.
(the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
    * **binary** (only two classes)
    * **categorical** (not only two class)
    * **input** (developing an Autoencoder system, both input and the output would probably be the same image)
* shuffle
    * Set True if you want to shuffle the order of the image that is being yielded, else set False
* seed
    * Random seed for applying random image augmentation and shuffling the order of the image
* [tutorial web page](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)
### keras earlystopping
```sh
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0
, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
```
* patience: 沒有進步的訓練輪數，在這之後訓練就會被停止
### keras checkpoint
```sh
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
save_best_only=False, save_weights_only=False, mode='auto', period=1)
```
* filepath: 字符串，保存模型的路径。
* monitor: 被监测的数据。
* verbose: 详细信息模式，0 或者 1 。  git config --global push.default matching

To squelch this message and adopt the new behavior now, use:

  git config --global push.default simple

When push.default is set to 'matching', git will push local branches
to the remote branches that already exist with the same name.

Since Git 2.0, Git defaults to the more conservative 'simple'
behavior, which only pushes the current branch to the corresponding
remote branch that 'git pull' uses to update the current branch.

See 'git help config' and search for 'push.default' for further information.
(the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
* save_best_only: 如果 save_best_only=True， 被监测数据的最佳模型就不会被覆盖。
### Keras Data augmentation
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

* [reference web](https://blog.keras.io/buil  git config --global push.default matching
## large file git handle
* https://stackoverflow.com/questions/19573031/cant-push-to-github-because-of-large-file-which-i-already-deleted


