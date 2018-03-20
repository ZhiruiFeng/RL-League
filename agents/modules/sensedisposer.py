import tensorflow as tf

class SenseDisposer:
    """
    Transfer the observation into states.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("sense_disposer"):
            self.input_senses = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_senses)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, raw_sense):
        """
        Return:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, {self.input_senses: raw_sense})
