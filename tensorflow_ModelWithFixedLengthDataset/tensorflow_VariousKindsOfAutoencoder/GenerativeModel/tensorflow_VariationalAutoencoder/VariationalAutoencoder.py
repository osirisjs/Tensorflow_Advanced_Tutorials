import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


# evaluate the data
def show_image(model_name, generated_image, column_size=10, row_size=10):
    print("show image")
    '''generator image visualization'''
    fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig_g.suptitle('MNIST_generator')
    for j in range(row_size):
        for i in range(column_size):
            ax_g[j][i].grid(False)
            ax_g[j][i].set_axis_off()
            ax_g[j][i].imshow(generated_image[i + j * column_size].reshape((28, 28)), cmap='gray')
    fig_g.savefig("{}_generator.png".format(model_name))
    plt.show()


def model(TEST=True, latent_number=16, model_name="Autoencoder", optimizer_selection="Adam",
          learning_rate=0.001, training_epochs=100,
          batch_size=128, display_step=10, batch_norm=True):
    mnist = input_data.read_data_sets("", one_hot=False)

    if batch_norm == True:
        model_name = "batch_norm_" + model_name

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    # stride? -> [1, 2, 2, 1] = [one image, width, height, one channel]
    def conv2d(input, weight_shape='', bias_shape='', strides=[1, 1, 1, 1], padding="VALID"):
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        bias_init = tf.constant_initializer(value=0)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(0.00001, dtype=tf.float32)
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))

        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding)

        if batch_norm:
            return tf.layers.batch_normalization(tf.nn.bias_add(conv_out, b), training=not TEST)
        else:
            return tf.nn.bias_add(conv_out, b)

    def conv2d_transpose(input, output_shape='', weight_shape='', bias_shape='', strides=[1, 1, 1, 1], padding="VALID"):
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        bias_init = tf.constant_initializer(value=0)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(0.00001, dtype=tf.float32)
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)
        if batch_norm:
            return tf.layers.batch_normalization(tf.nn.bias_add(conv_out, b), training=not TEST)
        else:
            return tf.nn.bias_add(conv_out, b)

    # ksize, strides? -> [1, 2, 2, 1] = [one image, width, height, one channel]
    # pooling을 할때, 각 batch 에 대해 한 채널에 대해서 하니까, 1, 1,로 설정해준것.
    def pooling(input, type="avg", k=2, padding='VALID'):
        if type == "max":
            return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)
        else:
            return tf.nn.avg_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

    def layer(input, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=0.01)
        bias_init = tf.random_normal_initializer(stddev=0.01)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(0.00001, dtype=tf.float32)
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        if batch_norm:
            return tf.layers.batch_normalization(tf.matmul(input, w) + b, training=not TEST)
        else:
            return tf.matmul(input, w) + b

    def inference(x, target, latent_number):

        if model_name == "Autoencoder" or model_name == "batch_norm_Autoencoder":
            with tf.variable_scope("encoder"):
                with tf.variable_scope("fully1"):
                    fully_1 = tf.nn.relu(layer(tf.reshape(x, (-1, 784)), [784, 256], [256]))
                with tf.variable_scope("fully2"):
                    fully_2 = tf.nn.relu(layer(fully_1, [256, 128], [128]))
                with tf.variable_scope("fully3"):
                    fully_3 = tf.nn.relu(layer(fully_2, [128, 64], [64]))

            with tf.variable_scope("mean_variance"):
                # 활성화 함수 쓰면 안된다.
                encoder_output = layer(fully_3, [64, latent_number * 2], [latent_number * 2])

                '''key point1
                reparametrization trick'''
                mu, log_var = tf.split(encoder_output, [latent_number, latent_number], axis=1)
                zero_mean_gaussian = tf.random_normal([tf.shape(x)[0], latent_number], mean=0.0, stddev=1.0)
                std = tf.exp(0.5 * log_var)  # 양수로 만들기 위함
                latent_variable = tf.concat([mu + std * zero_mean_gaussian, tf.tile(tf.reshape(target, (-1, 1)),
                                                                                    [1, latent_number])], axis=1)

            # 학습이 다 완료된 후에는 아래의 decoder만 사용하면 된다.
            with tf.variable_scope("decoder"):
                with tf.variable_scope("fully1"):
                    fully_4 = tf.nn.relu(layer(latent_variable, [latent_number * 2, 64], [64]))
                with tf.variable_scope("fully2"):
                    fully_5 = tf.nn.relu(layer(fully_4, [64, 128], [128]))
                with tf.variable_scope("fully3"):
                    fully_6 = tf.nn.relu(layer(fully_5, [128, 256], [256]))
                with tf.variable_scope("output"):
                    decoder_output = tf.nn.sigmoid(layer(fully_6, [256, 784], [784]))

            return latent_variable, encoder_output, decoder_output

        elif model_name == 'Convolution_Autoencoder' or model_name == "batch_norm_Convolution_Autoencoder":
            with tf.variable_scope("encoder"):
                with tf.variable_scope("conv_1"):
                    conv_1 = tf.nn.relu(
                        conv2d(x, weight_shape=[5, 5, 1, 32], bias_shape=[32], strides=[1, 1, 1, 1], padding="VALID"))
                    # result -> batch_size, 24, 24, 32

                with tf.variable_scope("conv_2"):
                    conv_2 = tf.nn.relu(
                        conv2d(conv_1, weight_shape=[5, 5, 32, 32], bias_shape=[32], strides=[1, 1, 1, 1],
                               padding="VALID"))
                    # result -> batch_size, 20, 20, 32

            with tf.variable_scope("mean_variance"):
                # 활성화 함수 쓰면 안된다.
                encoder_output = conv2d(conv_2, weight_shape=[5, 5, 32, latent_number * 2],
                                        bias_shape=[latent_number * 2],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID")
                # result -> batch_size, 16, 16, 32
                '''key point1
                reparametrization trick'''
                mu, log_var = tf.split(encoder_output, [latent_number, latent_number], axis=-1)
                zero_mean_gaussian = tf.random_normal(
                    [tf.shape(x)[0], tf.shape(encoder_output)[1], tf.shape(encoder_output)[2], latent_number], \
                    mean=0.0, stddev=1.0)
                std = tf.exp(0.5 * log_var)  # 양수로 만들기 위함
                latent_variable = tf.concat([mu + std * zero_mean_gaussian, \
                                             tf.tile(tf.reshape(target, (-1, 1, 1, 1)), [1, tf.shape(encoder_output)[1],
                                                                                         tf.shape(encoder_output)[2],
                                                                                         latent_number])], axis=-1)

            # 학습이 다 완료된 후에는 아래의 decoder만 사용하면 된다.

            with tf.variable_scope("decoder"):
                with tf.variable_scope("trans_conv_1"):
                    conv_3 = tf.nn.relu(
                        conv2d_transpose(latent_variable, output_shape=tf.shape(conv_2),
                                         weight_shape=[5, 5, 32, latent_number * 2],
                                         bias_shape=[32], strides=[1, 1, 1, 1], padding="VALID"))
                    # result -> batch_size, 20, 20, 32

                with tf.variable_scope("trans_conv_2"):
                    conv_4 = tf.nn.relu(
                        conv2d_transpose(conv_3, output_shape=tf.shape(conv_1), weight_shape=[5, 5, 32, 32],
                                         bias_shape=[32], strides=[1, 1, 1, 1], padding="VALID"))
                    # result -> batch_size, 24, 24, 32

                with tf.variable_scope("output"):
                    decoder_output = tf.nn.sigmoid(
                        conv2d_transpose(conv_4, output_shape=tf.shape(x), weight_shape=[5, 5, 1, 32],
                                         bias_shape=[1],
                                         strides=[1, 1, 1, 1], padding="VALID"))
                    # result -> batch_size, 28, 28, 1

            return latent_variable, encoder_output, decoder_output

    def evaluate(output, x):
        with tf.variable_scope("validation"):
            tf.summary.image('input_image', tf.reshape(x, [-1, 28, 28, 1]), max_outputs=5)
            tf.summary.image('output_image', tf.reshape(output, [-1, 28, 28, 1]), max_outputs=5)

            if model_name == 'Convolution_Autoencoder' or model_name == "batch_norm_Convolution_Autoencoder":
                l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), axis=[1, 2, 3]))
            elif model_name == "Autoencoder" or model_name == "batch_norm_Autoencoder":
                l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, tf.reshape(x, (-1, 784)))), axis=1))

            val_loss = tf.reduce_mean(l2)
            tf.summary.scalar('val_cost', val_loss)
            return val_loss

    def crossentropy(output, x):
        if model_name == "Autoencoder" or model_name == "batch_norm_Autoencoder":
            transformed_x = tf.reshape(x, (-1, 784))
            train_loss = tf.reduce_sum(transformed_x * tf.log(output + 1e-12) + \
                                       (1 - transformed_x) * tf.log(1 - output + 1e-12), axis=1)
        elif model_name == 'Convolution_Autoencoder' or model_name == "batch_norm_Convolution_Autoencoder":
            train_loss = tf.reduce_sum(x * tf.log(output + 1e-12) + \
                                       (1 - x) * tf.log(1 - output + 1e-12), axis=[1, 2, 3])
        return -tf.reduce_mean(train_loss)

    '''key point2'''

    def latentloss(encoder_output):
        if model_name == "Autoencoder" or model_name == "batch_norm_Autoencoder":
            mu, log_var = tf.split(encoder_output, [latent_number, latent_number], axis=1)
            train_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + mu * mu - 1 - log_var, axis=1)
        elif model_name == 'Convolution_Autoencoder' or model_name == "batch_norm_Convolution_Autoencoder":
            mu, log_var = tf.split(encoder_output, [latent_number, latent_number], axis=-1)
            train_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + mu * mu - 1 - log_var, axis=[1, 2, 3])
        return tf.reduce_mean(train_loss)

    def training(cost):
        tf.summary.scalar("train_cost", cost)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(cost)
        return train_operation

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        with tf.name_scope("feed_dict"):
            x = tf.placeholder("float", [None, 28, 28, 1])
            target = tf.placeholder("float", [None])
        with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope("inference"):
                latent_variable, encoder_output, decoder_output = inference(x, target, latent_number)
            # scope.reuse_variables()

        # Adam optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
        with tf.name_scope("saver"):
            saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
            # set으로 중복 제거 하고, 다시 list로 바꾼다.
            saver_generator = tf.train.Saver(var_list=list(set(np.concatenate(( \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_variables/decoder'), \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/decoder')), \
                axis=0))), max_to_keep=3)

        if not TEST:
            # Variational Auotoencoder Loss
            with tf.name_scope("loss"):
                cost = crossentropy(decoder_output, x) + latentloss(encoder_output)
            with tf.name_scope("trainer"):
                train_operation = training(cost)
            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

        with tf.name_scope("evaluation"):
            evaluate_operation = evaluate(decoder_output, x)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=JG_Graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_all = tf.train.get_checkpoint_state(os.path.join('model', model_name, 'all'))
        ckpt_generator = tf.train.get_checkpoint_state(os.path.join('model', model_name, 'generator'))
        if (ckpt_all and tf.train.checkpoint_exists(ckpt_all.model_checkpoint_path)) \
                or (ckpt_generator and tf.train.checkpoint_exists(ckpt_generator.model_checkpoint_path)):
            if not TEST:
                print("all variable retored except for optimizer parameter")
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_all.model_checkpoint_path)))
                saver_all.restore(sess, ckpt_all.model_checkpoint_path)
                # shutil.rmtree("model/{}/".format(model_name))
            else:
                print("generator variable retored except for optimizer parameter")
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_generator.model_checkpoint_path)))
                saver_generator.restore(sess, ckpt_generator.model_checkpoint_path)
                # saver_all.restore(sess, ckpt_all.model_checkpoint_path)

        if not TEST:

            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)

            for epoch in tqdm(range(1, training_epochs + 1)):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / batch_size)
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x: mbatch_x.reshape((-1, 28, 28, 1)), target: mbatch_y}
                    _, minibatch_cost = sess.run([train_operation, cost], feed_dict=feed_dict)
                    avg_cost += (minibatch_cost / total_batch)

                print("cost : {}".format(avg_cost))
                if epoch % display_step == 0:
                    val_feed_dict = {x: mnist.validation.images[:1000].reshape(
                        (-1, 28, 28, 1)),
                        target: mnist.validation.labels[:1000]}  # GPU 메모리 인해 mnist.test.images[:1000], 여기서 1000이다.
                    val_cost, summary_str = sess.run([evaluate_operation, summary_operation],
                                                     feed_dict=val_feed_dict)
                    print("Validation L2 cost : {}".format(val_cost))
                    summary_writer.add_summary(summary_str, global_step=epoch)

                    save_all_model_path = os.path.join('model', model_name, 'all/')
                    save_generator_model_path = os.path.join('model', model_name, 'generator/')

                    if not os.path.exists(save_all_model_path):
                        os.makedirs(save_all_model_path)
                    if not os.path.exists(save_generator_model_path):
                        os.makedirs(save_generator_model_path)

                    saver_all.save(sess, save_all_model_path, global_step=epoch,
                                   write_meta_graph=False)
                    saver_generator.save(sess, save_generator_model_path,
                                         global_step=epoch,
                                         write_meta_graph=False)

            print("Optimization Finished!")

        # batch_norm=True 일 때, 이동평균 사용
        if TEST:
            column_size = 10
            row_size = 10
            if model_name == "Autoencoder" or model_name == "batch_norm_Autoencoder":
                target = np.tile(np.tile(np.arange(start=0, stop=column_size), row_size).reshape((-1, 1)),
                                 (1, latent_number))
                feed_dict = {latent_variable: np.concatenate(
                    (np.random.normal(loc=0.0, scale=1.0, size=(column_size * row_size, latent_number)), target) \
                    , axis=1)}
            elif model_name == 'Convolution_Autoencoder' or model_name == "batch_norm_Convolution_Autoencoder":
                '''x: np.zeros(shape=(column_size * row_size, 28, 28, 1))는 conv2d_transpose 에서 output_shape를 위해 필요하다.
                위 코드에서 이런식으로 (output_shape=tf.shape(conv_1)) 으로 사용하고 있기 때문이다.
                '''
                target = np.tile(np.tile(np.arange(start=0, stop=column_size), row_size).reshape((-1, 1, 1, 1)),
                                 (1, np.shape(encoder_output)[1], np.shape(encoder_output)[2], latent_number))
                feed_dict = {x: np.zeros(shape=(column_size * row_size, 28, 28, 1)),
                             latent_variable: np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=(
                                 column_size * row_size, np.shape(encoder_output)[1], np.shape(encoder_output)[2],
                                 latent_number)), target), axis=-1)}
            generated_image = sess.run(decoder_output, feed_dict=feed_dict)
            show_image(model_name, generated_image, column_size=column_size, row_size=row_size)


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    # model_name = "Convolution_Autoencoder" or "Autoencoder"
    # latent_number는 2의 배수인 양수여야 한다.
    model(TEST=True, latent_number=16, model_name="Autoencoder", optimizer_selection="Adam", \
          learning_rate=0.001, training_epochs=300, batch_size=512, display_step=1, batch_norm=True)

else:
    print("model imported")
