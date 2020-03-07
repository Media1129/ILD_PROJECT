# GITHUB_ILD
## Run the Project
```sh
source activate tensorflow
python resnet_train.py
python resnet_predict.py ../model_output/001_train.h5
```
## evaluate vs training accuracy problem
* [reference_1 webpage](https://stackoverflow.com/questions/47157526/resnet-100-accuracy-during-training-but-33-prediction-accuracy-with-the-same)
* [reference_2 webpage](https://github.com/keras-team/keras/issues/8411)
* above reference say the the reason why the evaluate different from training accuracy,because of 
    * the batch normalization layers.
    * In training phase, the batch is normalized w.r.t. its mean and variance. 
    * However, in testing phase, the batch is normalized w.r.t. the moving average of previously observed mean and variance.
* my try (Use easy dataset cat_and_dog train)
    * model.evaluate_generator and  training accuracy are same(0.99)
    * model.evaluate_generator and previous predict outcome are same
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
## second train
### Train
* ![](https://i.imgur.com/8dyP2ow.jpg)
* **accuracy:0.77**
### Test
* ![](https://i.imgur.com/XagCIhB.png)
* **accuracy:0.64**
    * 002_train.h5
## -----------------------------------------------------------
## third train
### Train
* ![](https://i.imgur.com/spUaoeu.jpg)
* **accuracy:0.948**
### Test
* ![](https://i.imgur.com/Tb2sIxq.jpg)
* **accuracy:0.75**
    * 005_train.h5
## -----------------------------------------------------------
## four train 
### Train
* ![](https://i.imgur.com/3sxXteA.jpg)
* **accuracy:0.806**
### Test
* ![](https://i.imgur.com/YZYo2Y7.jpg)
* **accuracy:0.71**
    * 014_train.h5
