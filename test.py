import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t1 = tf.random_uniform(dtype=tf.float32, minval=0, maxval=2, shape=[5])
    t2 = tf.random_uniform(dtype=tf.int32, minval=0, maxval=2, shape=[5])
    m = tf.confusion_matrix(labels=t1, predictions=t2)
    t3 = tf.argmax(t1, axis=0)

    print(tf.count_nonzero(tf.cast(tf.equal(t1, 1.0), tf.int32)).eval())
