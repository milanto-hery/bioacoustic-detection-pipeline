"""
networks.py
Standardized CNN architectures for bioacoustic classification using a Model Factory approach.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.regularizers import l2


def build_custom_cnn(input_shape, num_classes=2, dropout_rate=0.3):
    """
    Builds a lightweight, multi-layer CNN optimized for grayscale spectrograms (1-channel).
    Useful for low-resource environments and fast experimentation.
    """
    model = models.Sequential(name="custom_cnn")
    model.add(layers.Input(shape=input_shape))

    # Feature extraction blocks
    model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _adapt_input_for_pretrained(inputs):
    """
    Ensures input has 3 channels for pre-trained models.
    Replicates 1-channel grayscale inputs across RGB channels.
    """
    if inputs.shape[-1] == 1:
        return tf.tile(inputs, [1, 1, 1, 3])
    return inputs


def build_mobilenet_v2(input_shape, num_classes=2, dropout_rate=0.3, fine_tune=False):
    """
    MobileNetV2: Highly efficient architecture optimized for mobile and edge devices.
    Good balance between speed and accuracy for bioacoustic tasks.
    """
    inputs = layers.Input(shape=input_shape)
    x = _adapt_input_for_pretrained(inputs)

    base_model = MobileNetV2(
        include_top=False, 
        weights="imagenet", 
        input_tensor=x
    )
    
    # Freeze base model if fine_tune is False
    base_model.trainable = fine_tune
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=out, name="mobilenet_v2")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_efficientnet_b0(input_shape, num_classes=2, dropout_rate=0.3, fine_tune=False):
    """
    EfficientNetB0: Uses compound scaling to achieve state-of-the-art efficiency.
    Excellent choice for maximizing accuracy with a limited compute budget.
    """
    inputs = layers.Input(shape=input_shape)
    x = _adapt_input_for_pretrained(inputs)

    base_model = EfficientNetB0(
        include_top=False, 
        weights="imagenet", 
        input_tensor=x
    )
    
    # Freeze base model if fine_tune is False
    base_model.trainable = fine_tune
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=out, name="efficientnet_b0")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_model(model_name, input_shape, num_classes, **kwargs):
    """
    Model Factory Entry Point. Returns the compiled model by name.
    
    Args:
        model_name: Name of the architecture (custom, mobilenet_v2, efficientnet_b0)
        input_shape: Shape of the input spectrogram (H, W, C)
        num_classes: Number of classification categories
        **kwargs: Additional parameters like dropout_rate and fine_tune
    """
    name = model_name.lower()
    
    if name in ["custom", "baseline", "custom_cnn"]:
        # CustomCNN only uses dropout_rate; fine_tune is not applicable
        cnn_kwargs = {k: v for k, v in kwargs.items() if k in ["dropout_rate"]}
        return build_custom_cnn(input_shape, num_classes, **cnn_kwargs)
    elif name == "mobilenet_v2":
        # Pre-trained models support dropout_rate and fine_tune
        pretrained_kwargs = {k: v for k, v in kwargs.items() if k in ["dropout_rate", "fine_tune"]}
        return build_mobilenet_v2(input_shape, num_classes, **pretrained_kwargs)
    elif name == "efficientnet_b0":
        pretrained_kwargs = {k: v for k, v in kwargs.items() if k in ["dropout_rate", "fine_tune"]}
        return build_efficientnet_b0(input_shape, num_classes, **pretrained_kwargs)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Valid options: 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0'"
        )
