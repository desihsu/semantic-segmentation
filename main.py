import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import tests


print('TensorFlow Version: {}'.format(tf.__version__))
assert (LooseVersion(tf.__version__) >= LooseVersion('1.0'),
    'Please use TensorFlow version 1.0 or newer.')

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')
    
    return image_input, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    decode1 = tf.layers.conv2d_transpose(layer7, num_classes, 4, 2, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    decode2 = tf.add(decode1, layer4)
    decode3 = tf.layers.conv2d_transpose(decode2, num_classes, 4, 2, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    decode4 = tf.add(decode3, layer3)
    output = tf.layers.conv2d_transpose(decode4, num_classes, 16, 8, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    sess.run(tf.global_variables_initializer())
    print('Training...')
    print()

    for epoch in range(epochs):
        print('EPOCH {} ...'.format(epoch+1))

        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: 0.5, learning_rate: 0.0005})
            print('Loss: {}'.format(loss))
            
        print()

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    tests.test_for_kitti_dataset(data_dir)
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # Needs a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        epochs = 50
        batch_size = 5

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
