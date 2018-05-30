import tensorflow as tf

#创建一个常量op
m1 = tf.constant([[3,3]])
#创建一个常量op
m2 = tf.constant([[2],[3]])
#创建一个矩阵乘法 把m1  m2传入
product = tf.matmul(m1,m2)
print(product)  #Tensor("MatMul:0", shape=(1, 1), dtype=int32)


#定义一个会话 启动默认图
sess = tf.Session()
#调用sess的run方法 执行乘法op
#run(product)方法会出大图中的3个op
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result = sess.run(product)
    print(result)