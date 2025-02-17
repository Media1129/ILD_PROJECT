import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense,Dropout
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.preprocessing import image


DATA_DIR = '../ILD_DATA/cut_data/first_cut'
# DATA_DIR = "../cat_dog_data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
# BATCH_SIZE = 32
BATCH_SIZE = 8




if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    # val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    
    model = keras.applications.resnet50.ResNet50(weights=None)

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=True
    last = model.layers[-1].output
    # x = Dense(len(classes), activation="softmax")(last)
    
    # one = Dense(4,activation="sigmoid")(last)
    # mid = Dropout(0.5)(one)
    # x = Dense(len(classes), activation="softmax")(one)
    
    one = Dense(32,activation="relu")(last)
    mid = Dropout(0.5)(one)
    two = Dense(16,activation="relu")(one)
    final = Dropout(0.5)(two)
    x = Dense(len(classes), activation="softmax")(two)
    
    
    finetuned_model = Model(model.input, x) 
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    

    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    # early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('../model_output/resnet50_best.h5', verbose=1, save_best_only=True)

    # finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=200, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=4, callbacks=[checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    # finetuned_model.save('resnet50_final.h5')