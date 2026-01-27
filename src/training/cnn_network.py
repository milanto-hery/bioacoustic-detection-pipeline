"""
cnn_network.py
Model builders. Keep a simple baseline CNN and wrappers for pretrained backbones (optional).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG19
from keras.regularizers import l2


def build_baseline_cnn(input_shape, num_classes=2, dropout_rate=0.3):
    model = models.Sequential(name="baseline_cnn")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_resnet50(input_shape, num_classes=2):
    base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mobilenet_v2(input_shape, num_classes=2):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_vgg19(input_shape, num_classes=2):
    base = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
