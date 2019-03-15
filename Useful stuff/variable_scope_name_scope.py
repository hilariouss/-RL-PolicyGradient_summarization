import tensorflow as tf
import numpy as np

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

a = tf.Variable(3, name="var_a", dtype=tf.int32)
b = tf.Variable(2, name="var_b", dtype=tf.int32)

with tf.variable_scope("var_scope"):
    c = tf.Variable(2, name='c', dtype=tf.int32)
    with tf.name_scope("name_scope"):
        v1 = tf.get_variable("var_a", [1], dtype=tf.int32)
        v2 = tf.Variable(3, name="var_b", dtype=tf.int32)

print(c) # var_scope/c:0

print(v1) # name scope "name_scope" 생략 : var_scope/var_a:0
print(v2) # name scope "name_scope" 생략 x : var_scope/name_scope/var_b:0

print(a) # var_a:0
print(b) # var_b:0

# 즉, with 구문으로 tf.variable_scope내의 이름과 외부의 이름은 겹치지 않음.
# 이름 영역이 다르게 구별되어 이름이 같아도 다른 변수로 취급됨. (a와 v1, b와 v2)
# 또한, name_scope내에서 get_variable()함수를 사용하면, 해당 name_scope의 이름영역이 무시됨.