GITHUB_ILD
===
:::info
* **本周進度**
    - 分析模型預測好壞的原因
    - 利用切出肺的CT照片預測
:::


:closed_book: Run the Project
-- 
```sh
source activate tensorflow
python resnet_train.py
python resnet_predict.py ../model_output/001_train.h5
```

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

:closed_book:  First Train
-- 
### Train
<!-- * ![](https://i.imgur.com/8dyP2ow.jpg) -->
* ![](https://i.imgur.com/ETIdnfa.jpg)
* **accuracy:1.0**
### Test
<!-- * ![](https://i.imgur.com/XagCIhB.png) -->
* ![](https://i.imgur.com/bVuZf7i.jpg)
* **accuracy:0.85**
* **Analysis the result**
    * patient_10 
        * 前20張沒有那麼明顯
        * 後面十張預測錯是因為已經不是肺了
    * patient_11
        * 前20張CT預測錯 
        * 看不太出來原因
    * 019_train.h5

:closed_book:  Second Train
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
* **Analysis the result**
    * patient_1
        * all error 
        * CT照片相較其他病患網狀沒有那麼嚴重
    * 017_train.h5

:closed_book: Third Train 
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
    * patient 1,2,7,12 錯很多
    * 可能是train沒有訓練到
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
