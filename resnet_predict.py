# import IPython
# import json, os, re, sys, time
# import numpy as np

# from keras import backend as K
# from keras.applications.imagenet_utils import preprocess_input
# from keras.models import load_model
# from keras.preprocessing import image


# def predict(img_path, model):
#     print('Generating predictions on image:', img_path)
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     preds = model.predict(x)
#     print(preds)
#     return preds

# if __name__ == '__main__':
#     model_path = sys.argv[1]
#     t0 = time.time()
#     model = load_model(model_path)
#     t1 = time.time()


#     for i in range(int(sys.argv[4])):
#         predict('data_store/ALL_'+sys.argv[2]+'/'+sys.argv[3]+'/'+sys.argv[3]+'_'+str(i+1)+'.jpg',model)
import IPython
import json, os, re, sys, time
import numpy as np

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
ILD_NUM = 0
NORMAL_NUM = 0

def predict(img_path, model):
    global ILD_NUM
    global NORMAL_NUM
    print('Generating predictions on image:', img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    print(preds)
    if preds[0][0] > preds[0][1]:
        ILD_NUM += 1
    else:
        NORMAL_NUM += 1
    print(ILD_NUM)
    print(NORMAL_NUM)
    print()
    return preds

if __name__ == '__main__':
    model_path = 'resnet50_best.h5'
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    
    ILD_TYPE = 'NORMAL'
    FOLDER = '15'
    picture_num = 117
    for i in range(picture_num):
        predict('data_store/ALL_'+ILD_TYPE+'/'+FOLDER+'/'+FOLDER+'_'+str(i+1)+'.jpg',model)
    print(ILD_NUM)
    print(NORMAL_NUM)