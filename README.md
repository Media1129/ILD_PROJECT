# GITHUB_ILD
:closed_book: Run the Project
-- 
```sh
source activate tensorflow
python resnet_train.py
python resnet_predict.py ../model_output/001_train.h5
```
:closed_book: evaluate vs training accuracy problem
--
- [reference_1 webpage](https://stackoverflow.com/questions/47157526/resnet-100-accuracy-during-training-but-33-prediction-accuracy-with-the-same)
- [reference_2 webpage](https://github.com/keras-team/keras/issues/8411)
- above reference say the the reason why the evaluate different from training accuracy,because of 
    - the batch normalization layers.
    - In training phase, the batch is normalized w.r.t. its mean and variance. 
    - However, in testing phase, the batch is normalized w.r.t. the moving average of previously observed mean and variance.
- my try (Use easy dataset cat_and_dog train)
    - model.evaluate_generator and  training accuracy are same(0.99)
    - model.evaluate_generator and previous predict outcome are same

:closed_book: DATA Description
--
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

:closed_book:  1_train
-- 
### Train
<!-- * ![](https://i.imgur.com/8dyP2ow.jpg) -->
* ![](https://i.imgur.com/ETIdnfa.jpg)
* **accuracy:1.0**
### Test
<!-- * ![](https://i.imgur.com/XagCIhB.png) -->
* ![](https://i.imgur.com/bVuZf7i.jpg)
* **accuracy:0.85**
    * patient_10 
        * 1-20 and 92-105 significant error
    * patient_11
        * 1-21 significant error
    * 019_train.h5

:closed_book:  2_train
--
### Train
<!-- * ![](https://i.imgur.com/aN35f6k.png) -->
<!-- * ![](https://i.imgur.com/spUaoeu.jpg) -->
* ![](https://i.imgur.com/lE5cn8d.jpg)
* **accuracy:1.0**
### Test
<!-- * ![](https://i.imgur.com/Tb2sIxq.jpg) -->
* ![](https://i.imgur.com/VH6V2bn.jpg)
* **accuracy:0.93**
    * patient_1
        * all error 
    * 017_train.h5

:closed_book: 3_train 
--
### Train
<!-- * ![](https://i.imgur.com/3sxXteA.jpg) -->
* ![](https://i.imgur.com/mNipVG6.jpg)

* **accuracy:1.0**
### Test
<!-- * ![](https://i.imgur.com/YZYo2Y7.jpg) -->
<!-- * ![](https://i.imgur.com/ohUkJnl.jpg) -->
* ![](https://i.imgur.com/TVDcSzI.jpg)
* **accuracy:0.69**
    * 018_train.h5

:closed_book:  CAM output 
--
* 觀察模型在意的特徵
### **With ILD**
* ![](https://i.imgur.com/HhZ76zZ.jpg)
* model predict:ILD
<!-- * 1_19 -->
* ![](https://i.imgur.com/8HvlLCN.jpg)
* model predict:ILD
<!-- * 5_23 -->
* ![](https://i.imgur.com/5xzgGXe.jpg)
* model predict:ILD
<!-- * 7_23 -->
* ![](https://i.imgur.com/7rlRkdu.jpg)
* model predict:ILD
<!-- * 9_23 -->
### **Without ILD**
* ![](https://i.imgur.com/BG5AQsX.jpg)
* model predict:Without ILD
<!-- * 11_35 -->
* ![](https://i.imgur.com/IKO9yPl.jpg)
* model predict:Without ILD
<!-- * 12_35 -->
* ![](https://i.imgur.com/Tnsc3Kh.jpg)
* model predict:Without ILD
<!-- * 13_15 -->
* ![](https://i.imgur.com/nWgJqbr.jpg)
* model predict:Without ILD
<!-- * 14_35 -->
