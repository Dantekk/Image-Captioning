import tensorflow as tf
import numpy as np

class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, brightness_delta, **kwargs):
        super(RandomBrightness, self).__init__(**kwargs)
        self.brightness_delta = brightness_delta

    def call(self, images, training=None):
        #if not training:
        #    return images

        brightness = np.random.uniform(self.brightness_delta[0], self.brightness_delta[1])

        images = tf.image.adjust_brightness(images, brightness)
        return images

class PowerLawTransform(tf.keras.layers.Layer):
    def __init__(self, gamma, **kwargs):
        super(PowerLawTransform, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, images, training=None):
        #if not training:
        #    return images

        gamma_value = np.random.uniform(self.gamma[0], self.gamma[1])

        images = tf.image.adjust_gamma(images, gamma_value)
        return images

class RandomSaturation(tf.keras.layers.Layer):
    def __init__(self, sat, **kwargs):
        super(RandomSaturation, self).__init__(**kwargs)
        self.sat = sat

    def call(self, images, training=None):
        #if not training:
        #    return images

        #sat_value = np.random.uniform(self.sat[0], self.sat[1])

        images = tf.image.random_saturation(images, self.sat[0], self.sat[1])
        return images

class RandomHue(tf.keras.layers.Layer):
    def __init__(self, hue, **kwargs):
        super(RandomHue, self).__init__(**kwargs)
        self.hue = hue

    def call(self, images, training=None):
        #if not training:
        #    return images

        #hue_value = np.random.uniform(self.hue[0], self.hue[1])

        images = tf.image.random_hue(images, self.hue[0], self.hue[1])
        return images
