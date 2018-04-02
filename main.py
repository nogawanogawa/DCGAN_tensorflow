from __future__ import division
import os
import numpy as np
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from glob import glob
from six.moves import xrange

from img_gen import plot
from ops import *

# 各種パラメータの設定
batch_size=100 # バッチのサイズ

input_height=28 # Discriminatorの入力サイズ(高さ)
input_width=28  # Discriminatorの入力サイズ(幅)

output_height=28 # Generatorの出力サイズ(高さ)
output_width=28  # Generatorの出力サイズ(高さ)

z_dim=100 # Generatorの入力サイズ（乱数）

gf_dim=64
df_dim=64

gfc_dim=1024
dfc_dim=1024

c_dim=1 #

total_epoch=2
sample_size = 10

dataset_name='mnist' # データセットの指定

checkpoint_dir=None
sample_dir=None

final_channel = c_dim

# Generatorのbatch normalizationの名前指定
g_batch_norm_0 = batch_norm(name = "g_batch_norm_0")
g_batch_norm_1 = batch_norm(name = "g_batch_norm_1")
g_batch_norm_2 = batch_norm(name = "g_batch_norm_2")
g_batch_norm_3 = batch_norm(name = "g_batch_norm_3")

# Discriminatorのbatch normalizationの名前指定
d_batch_norm_0 = batch_norm(name = "d_batch_norm_0")
d_batch_norm_1 = batch_norm(name = "d_batch_norm_1")
d_batch_norm_2 = batch_norm(name = "d_batch_norm_2")
d_batch_norm_3 = batch_norm(name = "d_batch_norm_3")

# 画像データの読み込み
if dataset_name == 'mnist':
    input_sample = input_data.read_data_sets("mnist/data/", one_hot=True)
    total_sample = input_sample.train.num_examples

n_input = input_height * input_width * final_channel

# placeholderの確保
z = tf.placeholder(tf.float32, [None, z_dim])
x = tf.placeholder(tf.float32, [None, n_input])

# deconvによる出力のサイズの算出
def deconv_size(size, stride):
    return int(math.ceil(float(size) / float(stride)))

g_h4, g_w4 = deconv_size(output_height, 2), deconv_size(output_width, 2)
g_h3, g_w3 = deconv_size(g_h4, 2), deconv_size(g_w4, 2)
g_h2, g_w2 = deconv_size(g_h3, 2), deconv_size(g_w3, 2)
g_h1, g_w1 = deconv_size(g_h2, 2), deconv_size(g_w2, 2)

# Generatorのセットアップ
def generator(z):
    # final result should be size of output_height, output_width
    # we suppose 4 conv layer and each strides with 2

    with tf.variable_scope('generator') as scope:
        noise = tf.nn.relu(linear(z, g_h1 * g_w1 * gfc_dim, 'g_lin_1'))
        noise_reshape = tf.reshape(noise, [-1, g_h1, g_w1, gfc_dim]) #
        noise_batch_relu = tf.nn.relu(g_batch_norm_0(noise_reshape)) #

        deconv_1 = deconv2d(noise_batch_relu, [batch_size, g_h2, g_w2, int(gfc_dim / 2)], name = 'g_deconv_1')
        deconv_1_batch_relu = tf.nn.relu(g_batch_norm_1(deconv_1))

        deconv_2 = deconv2d(deconv_1_batch_relu, [batch_size, g_h3, g_w3, int(gfc_dim / 4)], name = 'g_deconv_2')
        deconv_2_batch_relu = tf.nn.relu(g_batch_norm_2(deconv_2))

        deconv_3 = deconv2d(deconv_2_batch_relu, [batch_size, g_h4, g_w4, int(gfc_dim / 8)], name = 'g_deconv_3')
        deconv_3_batch_relu = tf.nn.relu(g_batch_norm_3(deconv_3))

        deconv_4 = deconv2d(deconv_3_batch_relu, [batch_size, output_height, output_width, final_channel], name = 'g_deconv_4')
        deconv_4_tanh = tf.nn.tanh(deconv_4)
        print('shape of generated tensor : {}'.format(deconv_4_tanh))

        return deconv_4_tanh

# Discriminatorのセットアップ
def discriminator(image, reuse = False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        # reshape the image if needed
        image = tf.reshape(image, [-1, input_height, input_height, final_channel])

        conv_1 = conv2d(image, df_dim, name = 'd_conv_1')
        conv_1_batch_lrelu = leakyReLU(d_batch_norm_0(conv_1))

        conv_2 = conv2d(conv_1_batch_lrelu, df_dim * 2, name = 'd_conv_2')
        conv_2_batch_lrelu = leakyReLU(d_batch_norm_1(conv_2))

        conv_3 = conv2d(conv_2_batch_lrelu, df_dim * 4, name = 'd_conv_3')
        conv_3_batch_lrelu = leakyReLU(d_batch_norm_2(conv_3))

        conv_4 = conv2d(conv_3_batch_lrelu, df_dim * 8, name = 'd_conv_4')
        conv_4_batch_lrelu = leakyReLU(d_batch_norm_3(conv_4))

        d_linear = linear(tf.reshape(conv_4_batch_lrelu, [-1, 8192]), 1, 'd_linear')

        return tf.nn.sigmoid(d_linear)


# Flowのセットアップ
G_z = generator(z)
D_G_z = discriminator(G_z)
D_x = discriminator(x, reuse = True)

# Lossの計算のセットアップ
loss_D = tf.reduce_mean(tf.log(D_x) + tf.log(1 - D_G_z))
loss_G = tf.reduce_mean(tf.log(D_G_z))

t_var = tf.trainable_variables()

g_vars = [v for v in t_var if 'g_' in v.name]
d_vars = [v for v in t_var if 'd_' in v.name]

saver = tf.train.Saver(max_to_keep = 1)

train_D = tf.train.AdamOptimizer(1e-4, beta1=0.5)\
                        .minimize(-loss_D, var_list = d_vars)
train_G = tf.train.AdamOptimizer(2e-4, beta1=0.5)\
                        .minimize(-loss_G, var_list = g_vars)

total_batch = int(total_sample/batch_size)
loss_val_D, loss_val_G = 0, 0

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 画像生成用ディレクトリの作成
if not os.path.exists('output/'):
    os.makedirs('output/')

# 乱数取得の関数定義
def get_noise(batch_size, n_noise):
    return np.random.uniform(-0.25, +0.25, size=(batch_size, n_noise))

j=0

# 学習プロセス開始
for epoch in range(total_epoch):
    for i in range(total_batch):

        # 1000回ごとに画像を保存
        if i % 10 == 0:
            Z_sample = get_noise(batch_size, z_dim)

            samples = sess.run(G_z, feed_dict={z: Z_sample})

            fig = plot(samples)
            plt.savefig('output/{}.png'.format(str(j).zfill(3)), bbox_inches='tight')
            j += 1
            plt.close(fig)

        # MNISTの教師データとノイズの取得
        batch_x, batch_y = input_sample.train.next_batch(batch_size)
        noise = get_noise(batch_size, z_dim)

        # DiscriminatorとGeneratorの実行
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict = {x: batch_x, z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict = {z: noise})

    if epoch == 0 or (epoch + 1) % 10 == 0:
        noise = get_noise(batch_size, z_dim)
        samples = sess.run(G_z, feed_dict={z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

    for i in range(sample_size):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)), cmap='Greys')

    plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
