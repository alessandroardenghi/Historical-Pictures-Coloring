import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow_examples.models.pix2pix import pix2pix


def unet_model_pretrained(output_channels:int, down_stack, up_stack , input_shape = [224, 224, 3]):
  inputs = tf.keras.layers.Input(shape=input_shape)

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def buildresnet_model(input_shape, OUTPUT_CLASSES):

    base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape = input_shape)

    # Use the activations of these layers
    layer_names_resnet = [
        'conv1_relu',       # 64x64
        'conv2_block3_out', # 32x32
        'conv3_block4_out', # 16x16
        'conv4_block6_out', # 8x8
        'conv5_block3_out', # 4x4
    ]

    base_resnet_outputs = [base_resnet.get_layer(name).output for name in layer_names_resnet]

    # Create the feature extraction model
    resnetdown_stack = tf.keras.Model(inputs=base_resnet.input, outputs=base_resnet_outputs)
    resnetdown_stack.trainable = False


    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]


    # Create the model
    return unet_model_pretrained(output_channels=OUTPUT_CLASSES, down_stack=resnetdown_stack, up_stack=up_stack, input_shape = input_shape)

def buildresnet_from_weights(input_shape, output_channels, finetuned, weight_path):
    model = buildresnet_model(input_shape, output_channels)
    if finetuned:
        for layer in model.layers:
            layer.trainable = True
    model.load_weights(weight_path)
    return model