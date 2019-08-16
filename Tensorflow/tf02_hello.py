import tensorflow as tf
print(tf.__version__)


hello = tf.constant('hello world')
sess = tf.Session()
print(sess.run(hello))