import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def xception_unet(
    input_shape,
    num_classes=1,
    filters=32,
    output_activation='sigmoid' # 'sigmoid' or 'softmax'
    ):
    
    inputs = tf.keras.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(filters, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for nb_filters in [filters*2, filters*4, filters*8]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(nb_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(nb_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(nb_filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for nb_filters in [filters*8, filters*4, filters*2, filters]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(nb_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(nb_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(nb_filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation=output_activation, padding="same")(x)

    # Define the model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
