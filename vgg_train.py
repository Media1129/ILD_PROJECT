import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense,Dropout
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.preprocessing import image


DATA_DIR = 'ild_data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
# BATCH_SIZE = 32
BATCH_SIZE = 16


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
    
    model = keras.applications.vgg16.VGG16()

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    # x = Dense(len(classes), activation="softmax")(last)
    
    one = Dense(1024,activation="sigmoid")(last)
    mid = Dropout(0.5)(one)
    two = Dense(256,activation="sigmoid")(mid)
    final = Dropout(0.5)(two)
    x = Dense(len(classes), activation="softmax")(final)
    
    
    finetuned_model = Model(model.input, x)
    # finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # finetuned_model.compile(optimizer=SGD(lr=0.0001,momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    # early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('output/vgg16_best.h5', verbose=1, save_best_only=True)

    # finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=1000, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=200, callbacks=[checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    
    finetuned_model.save('output/vgg16_final.h5')