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


def model(TEST=True, noise_size=100, targeting=True,
          optimizer_selection="Adam", learning_rate=0.001, training_epochs=100,
          batch_size=128, display_step=10, batch_norm=True):
    mnist = input_data.read_data_sets("", one_hot=True)

    if targeting == False:
        print("random generative GAN")
        model_name = "GeneralGAN"
    else:
        print("target generative GAN")
        model_name = "ConditionalGAN"

    if batch_norm == True:
        model_name = "batchnorm" + model_name

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    def layer(input, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=0.01)
        bias_init = tf.random_normal_initializer(stddev=0.01)
        if batch_norm:
            w = tf.get_variable("w", weight_shape, initializer=weight_init)
        else:
            weight_decay = tf.constant(0.000001, dtype=tf.float32)
            w = tf.get_variable("w", weight_shape, initializer=weight_init,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        if batch_norm:
            return tf.layers.batch_normalization(tf.matmul(input, w) + b, training=not TEST)
        else:
            return tf.matmul(input, w) + b

    def generator(noise=None, target=None):
        if targeting:
            noise = tf.concat([noise, target], axis=1)
        with tf.variable_scope("generator"):
            with tf.variable_scope("fully1"):
                fully_1 = tf.nn.relu(layer(noise, [np.shape(noise)[1], 256], [256]))
            with tf.variable_scope("fully2"):
                fully_2 = tf.nn.relu(layer(fully_1, [256, 512], [512]))
            with tf.variable_scope("output"):
                output = tf.nn.sigmoid(layer(fully_2, [512, 784], [784]))

        return output

    def discriminator(x=None, target=None):
        if targeting:
            x = tf.concat([x, target], axis=1)
        with tf.variable_scope("discriminator"):
            with tf.variable_scope("fully1"):
                fully_1 = tf.nn.relu(layer(x, [np.shape(x)[1], 500], [500]))
            with tf.variable_scope("fully2"):
                fully_2 = tf.nn.relu(layer(fully_1, [500, 100], [100]))
            with tf.variable_scope("output"):
                output = layer(fully_2, [100, 1], [1])
        return output

    def training(cost, var_list, scope=None):
        if scope == None:
            tf.summary.scalar("Discriminator Loss", cost)
        else:
            tf.summary.scalar("Generator Loss", cost)
        '''GAN 구현시 Batch Normalization을 쓸 때 주의할 점!!!
        #scope를 써줘야 한다. - 그냥 tf.get_collection(tf.GraphKeys.UPDATE_OPS) 이렇게 써버리면 
        shared_variables 아래에 있는 변수들을 다 업데이트 해야하므로 scope를 지정해줘야한다.
        - GAN의 경우 예)discriminator의 optimizer는 batch norm의 param 전체를 업데이트해야하고
                        generator의 optimizer는 batch_norm param의 generator 부분만 업데이트 해야 한다.   
        '''
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(cost, var_list=var_list)
        return train_operation

    def min_max_loss(logits=None, labels=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        with tf.name_scope("feed_dict"):
            x = tf.placeholder("float", [None, 784])
            target = tf.placeholder("float", [None, 10])
            z = tf.placeholder("float", [None, noise_size])
        with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope("generator"):
                G = generator(noise=z, target=target)
            with tf.name_scope("discriminator"):
                D_real = discriminator(x=x, target=target)
                # scope.reuse_variables()
                D_gene = discriminator(x=G, target=target)

        # Algorithjm
        var_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='shared_variables/discriminator')
        # set으로 중복 제거 하고, 다시 list로 바꾼다.
        var_G = list(set(np.concatenate(
            (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_variables/generator'),
             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/generator')),
            axis=0)))

        # Adam optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
        with tf.name_scope("saver"):
            saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
            saver_generator = tf.train.Saver(var_list=var_G, max_to_keep=3)

        if not TEST:
            # Algorithjm

            with tf.name_scope("Discriminator_loss"):
                # for discriminator
                D_Loss = min_max_loss(logits=D_real, labels=tf.ones_like(D_real)) + min_max_loss(logits=D_gene,
                                                                                                 labels=tf.zeros_like(
                                                                                                     D_gene))
            with tf.name_scope("Generator_loss"):
                # for generator
                G_Loss = min_max_loss(logits=D_gene, labels=tf.ones_like(D_gene))

            # Algorithjm
            with tf.name_scope("Discriminator_trainer"):
                D_train_op = training(D_Loss, var_D, scope=None)
            with tf.name_scope("Generator_trainer"):
                G_train_op = training(G_Loss, var_G, scope='shared_variables/generator')
            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=JG_Graph, config=config) as sess:
        print("initializing!!!")
        sess.run(tf.global_variables_initializer())
        ckpt_all = tf.train.get_checkpoint_state(os.path.join(model_name, 'All'))
        ckpt_generator = tf.train.get_checkpoint_state(os.path.join(model_name, 'Generator'))
        if (ckpt_all and tf.train.checkpoint_exists(ckpt_all.model_checkpoint_path)) \
                or (ckpt_generator and tf.train.checkpoint_exists(ckpt_generator.model_checkpoint_path)):
            if not TEST:
                print("all variable retored except for optimizer parameter")
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_all.model_checkpoint_path)))
                saver_all.restore(sess, ckpt_all.model_checkpoint_path)
            else:
                print("generator variable retored except for optimizer parameter")
                print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_generator.model_checkpoint_path)))
                saver_generator.restore(sess, ckpt_generator.model_checkpoint_path)

        if not TEST:
            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)

            for epoch in tqdm(range(1, training_epochs + 1)):

                Loss_D = 0.
                Loss_G = 0
                total_batch = int(mnist.train.num_examples / batch_size)
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, noise_size))
                    feed_dict_all = {x: mbatch_x, target: mbatch_y, z: noise}
                    feed_dict_Generator = {target: mbatch_y, z: noise}
                    _, Discriminator_Loss = sess.run([D_train_op, D_Loss], feed_dict=feed_dict_all)
                    _, Generator_Loss = sess.run([G_train_op, G_Loss], feed_dict=feed_dict_Generator)
                    Loss_D += (Discriminator_Loss / total_batch)
                    Loss_G += (Generator_Loss / total_batch)

                print("Discriminator Loss : {}, Generator Loss  : {}".format(Loss_D, Loss_G))

                if epoch % display_step == 0:
                    summary_str = sess.run(summary_operation, feed_dict=feed_dict_all)
                    summary_writer.add_summary(summary_str, global_step=epoch)

                    save_all_model_path = os.path.join(model_name, 'All/')
                    save_generator_model_path = os.path.join(model_name, 'Generator/')

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

        if TEST:
            column_size = 10
            row_size = 10
            feed_dict = {z: np.random.normal(loc=0.0, scale=1.0, size=(column_size * row_size, noise_size)),
                         target: np.tile(np.diag(np.ones(column_size)), (row_size, 1))}

            generated_image = sess.run(G, feed_dict=feed_dict)
            show_image(model_name, generated_image, column_size=column_size, row_size=row_size)


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    model(TEST=False, noise_size=128, targeting=True,
          optimizer_selection="Adam", learning_rate=0.0002, training_epochs=300,
          batch_size=128,
          # batch_norm을 쓰면 생성이 잘 안된다는..
          display_step=1, batch_norm=True)

else:
    print("model imported")
