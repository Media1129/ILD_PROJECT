import IPython
import json, os, re, sys, time
import numpy as np
import keras
import matplotlib
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical

ILD_NUM = 0
NORMAL_NUM = 0

def predict_cat_dog(folder_path,cat_num,dog_num): # predict the cat_dog dataset
    global ILD_NUM
    global NORMAL_NUM
    picture_num = [cat_num,dog_num]
    ILD = []
    NORMAL = []
    folder = ['cat','dog']

    for FOLDER in folder:
        if FOLDER == "cat":
            index = 0
        else:
            index = 1
        for i in range(picture_num[index]):
            pic_path = '../cat_dog_data/'+folder_path+'/'+FOLDER+'/'+str((i)).zfill(8)+'.jpg' 
            predict(pic_path,model)
        print("FOLDER=%s,ILD_NUM=%d,NORMAL_NUM=%d" % (FOLDER,ILD_NUM,NORMAL_NUM))
        ILD.append(ILD_NUM)
        NORMAL.append(NORMAL_NUM)
        ILD_NUM = 0
        NORMAL_NUM = 0

    labels = ['cat','dog']
    bar_graph_plot_two(labels,ILD,NORMAL,'Cat_Dog','PREDICT')

def evaluate_gen(DATA_DIR,folder_path,model): 
    EVA_DIR = os.path.join(DATA_DIR, folder_path)
    num_valid_samples = sum([len(files) for r, d, files in os.walk(EVA_DIR)])
    gen = keras.preprocessing.image.ImageDataGenerator()
    SIZE = (224, 224)
    BATCH_SIZE = 16
    batches = gen.flow_from_directory(EVA_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=1)
    c = model.evaluate_generator(generator=batches,steps=num_valid_samples) 
    print(c)


def predict(img_path, model):
    global ILD_NUM
    global NORMAL_NUM
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    # calculate the prediction type
    # print("pred1=%f pred2=%f" % (preds[0][0],preds[0][1]))
    if preds[0][0] > preds[0][1]:
        ILD_NUM += 1
        # print("ILD")
    else:
        NORMAL_NUM += 1
        # print("Normal")
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
    rects1 = ax.bar(x - width/2, lista, width, label='predict with ILD')
    rects2 = ax.bar(x + width/2, listb, width, label='predict without ILD')

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
def analyse_all(model,dir_path):
    global ILD_NUM
    global NORMAL_NUM
    # picture_num = [36,7,91,34,97,96,99,116,54,105,58,39,92,130,117]
    picture_num = [36,7,91,34,97,84,70,95,46,74,49,30,73,95,95]
    
    ILD = []
    NORMAL = []

    
    labels = [ str(i) for i in range(1,16)]
    title = "ALL "
    
    for FOLDER in range(1,16):
        for i in range(picture_num[FOLDER-1]):
            #before ten the ILD_TYPE is ILD
            ILD_TYPE = 'ILD' if FOLDER <= 10 else 'NORMAL'
            pic_path = dir_path+'/ALL_'+ILD_TYPE+'/'+str(FOLDER)+'/'+str(FOLDER)+'_'+str(i+1)+'.jpg' 
            # print(pic_path,end="  ")
            predict(pic_path,model)
        # print("\n\n\n")
        print("FOLDER=%d,ILD_NUM=%d,NORMAL_NUM=%d" % (FOLDER,ILD_NUM,NORMAL_NUM))
        ILD.append(ILD_NUM)
        NORMAL.append(NORMAL_NUM)
        ILD_NUM = 0
        NORMAL_NUM = 0
    bar_graph_plot_two(labels,ILD,NORMAL,title,'PREDICT')


def analyse(model):
    global ILD_NUM
    global NORMAL_NUM
    picture_num = [36,7,91,34,97,96,99,116,54,105,58,39,92,130,117]
    ILD = []
    NORMAL = []

    # third_train
    # folder = [2,3,6,7]
    # folder = [11,14,15]
    # folder = [1,4,5,8,9,10]
    # folder = [12,13]
    #second_train
    # folder = [1,4,5,8,9]
    # folder = [13,14,15]
    # folder = [2,3,6,7,10]
    # folder = [11,12]
    #four train
    # folder = [3,5,10]
    # folder = [11,13,14]
    # folder = [1,2,4,6,7,8,9]
    # folder = [12,15]
    

    labels = [ 'patient '+str(i) for i in folder]
    title = "With ILD"
    # title = "Without ILD"

    for FOLDER in folder:
        for i in range(picture_num[FOLDER-1]):
            #before ten the ILD_TYPE is ILD
            ILD_TYPE = 'ILD' if FOLDER <= 10 else 'NORMAL'
            pic_path = 'data_store/ALL_'+ILD_TYPE+'/'+str(FOLDER)+'/'+str(FOLDER)+'_'+str(i+1)+'.jpg' 
            # print(pic_path,end="  ")
            predict(pic_path,model)
        # print("\n\n\n")
        print("FOLDER=%d,ILD_NUM=%d,NORMAL_NUM=%d" % (FOLDER,ILD_NUM,NORMAL_NUM))
        ILD.append(ILD_NUM)
        NORMAL.append(NORMAL_NUM)
        ILD_NUM = 0
        NORMAL_NUM = 0

    bar_graph_plot_two(labels,ILD,NORMAL,title,'PREDICT')



if __name__ == '__main__':
    model_path = sys.argv[1]    #'resnet50_best.h5'
    print(model_path)
    model = load_model(model_path)

    # for ILD
    # evaluate_gen("second_train",'train',model) 
    # evaluate_gen("second_train",'test',model) 
    # analyse(model)
    analyse_all(model,'../ILD_DATA/cut_store/')
    # evaluate_gen('../ILD_DATA/cut_data/first_cut','train',model)
    # evaluate_gen('../ILD_DATA/cut_data/first_cut','test',model)



    # for cat_dog
    # evaluate_gen('train',"../cat_dog_data",model) 
    # evaluate_gen('valid',"../cat_dog_data",model) 
    # predict_cat_dog('train',171,177)
    # predict_cat_dog('valid',26,33)
