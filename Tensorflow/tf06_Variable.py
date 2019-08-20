import tensorflow as tf

# tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


W = tf.Variable([0.3], tf.float32)

## Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W))
sess.close()


# ## InteractiveSession
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# print(W.eval())
# sess.close()


## Session >> eval
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(W.eval(session=sess))


# ## with문 안에 일반 세션을 사용 >> InteractiveSession과 같음
# with tf.Session() as ss:
#     ss.run(tf.global_variables_initializer())

#     print(W.eval())
#     # print(ss.run(W))

