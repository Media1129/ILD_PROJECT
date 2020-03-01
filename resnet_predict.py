import IPython
import json, os, re, sys, time
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

ILD_NUM = 0
NORMAL_NUM = 0

def predict(img_path, model):
    global ILD_NUM
    global NORMAL_NUM
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    # calculate the prediction type
    if preds[0][0] > preds[0][1]:
        ILD_NUM += 1
    else:
        NORMAL_NUM += 1
    return preds

def bar_graph_plot_one(labels,lista,title_name,y_label_name):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, lista, width, label='ILD')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label_name)
    ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    fig.tight_layout()
    plt.show()
def bar_graph_plot_two(labels,lista,listb,title_name,y_label_name):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, lista, width, label='ILD')
    rects2 = ax.bar(x + width/2, listb, width, label='NORMAL')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label_name)
    ax.set_title(title_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 8),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # calculate the predict 
    model_path = 'resnet50_best.h5'
    model = load_model(model_path)
    picture_num = [36,7,91,34,97,96,99,116,54,105,58,39,92,130,117]
    ILD = []
    NORMAL = []
    # read the one-fifteen folder send to predict
    for FOLDER in range(1,16):
        for i in range(picture_num[FOLDER-1]):
            #before ten the ILD_TYPE is ILD
            ILD_TYPE = 'ILD' if FOLDER <= 10 else 'NORMAL'
            pic_path = 'data_store/ALL_'+ILD_TYPE+'/'+str(FOLDER)+'/'+str(FOLDER)+'_'+str(i+1)+'.jpg' 
            predict(pic_path,model)
        print("FOLDER=%d,ILD_NUM=%d,NORMAL_NUM=%d" % (FOLDER,ILD_NUM,NORMAL_NUM))
        ILD.append(ILD_NUM)
        NORMAL.append(NORMAL_NUM)
        ILD_NUM = 0
        NORMAL_NUM = 0

    labels = [ str(i) for i in range(1,16)]

    #plot the original data distribution
    # label_DATA = ['067(1)','120(2)','163(3)','192(4)','195(5)','196(6)','023(7) ','076(8)','034(9)','197(10)','006(11)','011(12)','016(13)','Normal_01(14)','Normal_02(15)']
    ILD_DATA = [36,7,91,34,97,96,99,116,54,105,0,0,0,0,0]
    NORMAL_DATA = [0,0,0,0,0,0,0,0,0,0,58,39,92,130,117]
    bar_graph_plot_two(labels,ILD_DATA,NORMAL_DATA,'Original DATA','DATA_NUM')
    #plot the bar graph by NORMAL and ILD list
    bar_graph_plot_two(labels,ILD,NORMAL,'ILD(ONE-TEN) VS NORMAL(ELEVEN-FIFTEEN)','PREDICT_NUM')
   

    
