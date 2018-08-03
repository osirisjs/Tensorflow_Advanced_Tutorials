import shutil

from Dataset import *


def visualize(model_name="Pix2PixConditionalGAN", named_images=None, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 이미지 y축 방향으로 붙이기
    image = np.hstack(named_images[1:])
    # 이미지 스케일 바꾸기(~1 ~ 1 -> 0~ 255)
    image = ((image + 1) * 127.5).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, '{}_{}.png'.format(model_name, named_images[0])), image)
    print("{}_{}.png saved in {} folder".format(model_name, named_images[0], save_path))


def model(TEST=False, AtoB=True, DB_name="maps", use_TFRecord=True, cycle_consistency_loss="L1",
          cycle_consistency_loss_weight=10,
          optimizer_selection="Adam", beta1=0.9, beta2=0.999,  # for Adam optimizer
          decay=0.999, momentum=0.9,  # for RMSProp optimizer
          use_identity_mapping=False,
          norm_selection="instance_norm", learning_rate=0.0002, training_epochs=200, batch_size=1, display_step=1,
          save_path="translated_image"):  # 학습 완료 후 변환된 이미지가 저장될 폴더 , AtoB=True -> AtoB_가 붙고, False -> BtoA_가 붙는다.

    print("CycleGAN")
    model_name = DB_name + "_" + "CycleGAN"

    if norm_selection == "instance_norm":
        model_name = "instancenorm_" + model_name

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    # 학습률 - 논문에서 100epoch 후에 선형적으로 줄인다고 했다.
    lr = tf.placeholder(dtype=tf.float32)

    def conv2d(input, weight_shape=None, bias_shape=None, norm_selection=None,
               strides=[1, 1, 1, 1], padding="VALID"):

        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        bias_init = tf.constant_initializer(value=0)

        weight_decay = tf.constant(0, dtype=tf.float32)
        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))

        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, w, strides=strides, padding=padding)

        if norm_selection == "instance_norm":
            return tf.contrib.layers.instance_norm(tf.nn.bias_add(conv_out, b))
        else:
            return tf.nn.bias_add(conv_out, b)

    def conv2d_transpose(input, output_shape=None, weight_shape=None, bias_shape=None, norm_selection=None,
                         strides=[1, 1, 1, 1], padding="VALID"):

        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        bias_init = tf.constant_initializer(value=0)
        weight_decay = tf.constant(0, dtype=tf.float32)

        w = tf.get_variable("w", weight_shape, initializer=weight_init,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        b = tf.get_variable("b", bias_shape, initializer=bias_init)

        conv_out = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding)

        if norm_selection == "instance_norm":
            return tf.contrib.layers.instance_norm(tf.nn.bias_add(conv_out, b))
        else:
            return tf.nn.bias_add(conv_out, b)

    # Residual Net
    def generator(images=None, name="G_generator"):

        '''encoder의 활성화 함수는 모두 leaky_relu이며, decoder의 활성화 함수는 모두 relu이다.
        encoder의 첫번째 층에는 batch_norm이 적용 안된다.

        총 16개의 층이다.
        '''

        with tf.variable_scope(name):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("conv1"):
                    conv1 = conv2d(images, weight_shape=(4, 4, np.shape(images)[-1], 64), bias_shape=(64),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 128, 128, 64)
                with tf.variable_scope("conv2"):
                    conv2 = conv2d(tf.nn.leaky_relu(conv1, alpha=0.2), weight_shape=(4, 4, 64, 128), bias_shape=(128),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 64, 64, 128)
                with tf.variable_scope("conv3"):
                    conv3 = conv2d(tf.nn.leaky_relu(conv2, alpha=0.2), weight_shape=(4, 4, 128, 256), bias_shape=(256),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 32, 32, 256)
                with tf.variable_scope("conv4"):
                    conv4 = conv2d(tf.nn.leaky_relu(conv3, alpha=0.2), weight_shape=(4, 4, 256, 512), bias_shape=(512),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 16, 16, 512)
                with tf.variable_scope("conv5"):
                    conv5 = conv2d(tf.nn.leaky_relu(conv4, alpha=0.2), weight_shape=(4, 4, 512, 512), bias_shape=(512),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 8, 8, 512)
                with tf.variable_scope("conv6"):
                    conv6 = conv2d(tf.nn.leaky_relu(conv5, alpha=0.2), weight_shape=(4, 4, 512, 512), bias_shape=(512),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 4, 4, 512)
                with tf.variable_scope("conv7"):
                    conv7 = conv2d(tf.nn.leaky_relu(conv6, alpha=0.2), weight_shape=(4, 4, 512, 512), bias_shape=(512),
                                   norm_selection=norm_selection,
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 2, 2, 512)
                with tf.variable_scope("conv8"):
                    conv8 = conv2d(tf.nn.leaky_relu(conv7, alpha=0.2), weight_shape=(4, 4, 512, 512), bias_shape=(512),
                                   strides=[1, 2, 2, 1], padding="SAME")
                    # result shape = (batch_size, 1, 1, 512)

            with tf.variable_scope("decoder"):
                with tf.variable_scope("trans_conv1"):
                    trans_conv1 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(conv8), output_shape=tf.shape(conv7), weight_shape=(4, 4, 512, 512),
                                         bias_shape=(512), norm_selection=norm_selection,
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=Dropout_rate)
                    # result shape = (batch_size, 2, 2, 512)
                    # 주의 : 활성화 함수 들어가기전의 encoder 요소를 concat 해줘야함
                    trans_conv1 = tf.concat([trans_conv1, conv7], axis=-1)
                    # result shape = (batch_size, 2, 2, 1024)

                with tf.variable_scope("trans_conv2"):
                    trans_conv2 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(trans_conv1), output_shape=tf.shape(conv6),
                                         weight_shape=(4, 4, 512, 1024),
                                         bias_shape=(512), norm_selection=norm_selection,
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=Dropout_rate)
                    trans_conv2 = tf.concat([trans_conv2, conv6], axis=-1)
                    # result shape = (batch_size, 4, 4, 1024)

                with tf.variable_scope("trans_conv3"):
                    trans_conv3 = tf.nn.dropout(
                        conv2d_transpose(tf.nn.relu(trans_conv2), output_shape=tf.shape(conv5),
                                         weight_shape=(4, 4, 512, 1024),
                                         bias_shape=(512), norm_selection=norm_selection,
                                         strides=[1, 2, 2, 1], padding="SAME"), keep_prob=Dropout_rate)
                    trans_conv3 = tf.concat([trans_conv3, conv5], axis=-1)
                    # result shape = (batch_size, 8, 8, 1024)

                with tf.variable_scope("trans_conv4"):
                    trans_conv4 = conv2d_transpose(tf.nn.relu(trans_conv3), output_shape=tf.shape(conv4),
                                                   weight_shape=(4, 4, 512, 1024),
                                                   bias_shape=(512), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv4 = tf.concat([trans_conv4, conv4], axis=-1)
                    # result shape = (batch_size, 16, 16, 1024)
                with tf.variable_scope("trans_conv5"):
                    trans_conv5 = conv2d_transpose(tf.nn.relu(trans_conv4), output_shape=tf.shape(conv3),
                                                   weight_shape=(4, 4, 256, 1024),
                                                   bias_shape=(256), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv5 = tf.concat([trans_conv5, conv3], axis=-1)
                    # result shape = (batch_size, 32, 32, 512)
                with tf.variable_scope("trans_conv6"):
                    trans_conv6 = conv2d_transpose(tf.nn.relu(trans_conv5), output_shape=tf.shape(conv2),
                                                   weight_shape=(4, 4, 128, 512),
                                                   bias_shape=(128), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv6 = tf.concat([trans_conv6, conv2], axis=-1)
                    # result shape = (batch_size, 64, 64, 256)
                with tf.variable_scope("trans_conv7"):
                    trans_conv7 = conv2d_transpose(tf.nn.relu(trans_conv6), output_shape=tf.shape(conv1),
                                                   weight_shape=(4, 4, 64, 256),
                                                   bias_shape=(64), norm_selection=norm_selection,
                                                   strides=[1, 2, 2, 1], padding="SAME")
                    trans_conv7 = tf.concat([trans_conv7, conv1], axis=-1)
                    # result shape = (batch_size, 128, 128, 128)
                with tf.variable_scope("trans_conv8"):
                    output = tf.nn.tanh(
                        conv2d_transpose(tf.nn.relu(trans_conv7), output_shape=tf.shape(target),
                                         weight_shape=(4, 4, 3, 128),
                                         bias_shape=(3),
                                         strides=[1, 2, 2, 1], padding="SAME"))
                    # result shape = (batch_size, 256, 256, 3)
        return output

    # PatchGAN
    def discriminator(input=None, name=None):

        '''discriminator의 활성화 함수는 모두 leaky_relu(slope = 0.2)이다.
        첫 번째 층에는 instance normalization 을 적용하지 않는다.
        왜 이런 구조를 사용? 아래의 구조 출력단의 ReceptiveField 크기를 구해보면 70이다.(ReceptiveFieldArithmetic/rf.py 에서 구해볼 수 있다.)'''
        with tf.variable_scope(name):
            with tf.variable_scope("conv1"):
                conv1 = tf.nn.leaky_relu(
                    conv2d(input, weight_shape=(4, 4, np.shape(input)[-1], 64), bias_shape=(64),
                           strides=[1, 2, 2, 1], padding="SAME"), alpha=0.2)
                # result shape = (batch_size, 128, 128, 64)
            with tf.variable_scope("conv2"):
                conv2 = tf.nn.leaky_relu(
                    conv2d(conv1, weight_shape=(4, 4, 64, 128), bias_shape=(128), norm_selection=norm_selection,
                           strides=[1, 2, 2, 1], padding="SAME"), alpha=0.2)
                # result shape = (batch_size, 64, 64, 128)
            with tf.variable_scope("conv3"):
                conv3 = conv2d(conv2, weight_shape=(4, 4, 128, 256), bias_shape=(256), norm_selection=norm_selection,
                               strides=[1, 2, 2, 1], padding="SAME")
                # result shape = (batch_size, 32, 32, 256)
                conv3 = tf.nn.leaky_relu(
                    tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0), alpha=0.2)
                # result shape = (batch_size, 34, 34, 256)
            with tf.variable_scope("conv4"):
                conv4 = conv2d(conv3, weight_shape=(4, 4, 256, 512), bias_shape=(512), norm_selection=norm_selection,
                               strides=[1, 1, 1, 1], padding="VALID")
                # result shape = (batch_size, 31, 31, 256)
                conv4 = tf.nn.leaky_relu(
                    tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0), alpha=0.2)
                # result shape = (batch_size, 33, 33, 512)
            with tf.variable_scope("output"):
                output = conv2d(conv4, weight_shape=(4, 4, 512, 1), bias_shape=(1),
                                strides=[1, 1, 1, 1], padding="VALID")
                # result shape = (batch_size, 30, 30, 1)
            return tf.nn.sigmoid(output)

    def training(cost, var_list, scope=None):

        tf.summary.scalar(scope, cost)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            if optimizer_selection == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
            elif optimizer_selection == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay, momentum=momentum)
            elif optimizer_selection == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_operation = optimizer.minimize(cost, var_list=var_list)
        return train_operation

    if not TEST:
    # print(tf.get_default_graph()) #기본그래프이다.
        JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

            # 데이터 전처리
            dataset = Dataset(DB_name=DB_name, AtoB=AtoB, batch_size=batch_size, use_TFRecord=use_TFRecord,
                              use_TrainDataset=not TEST)
            iterator, next_batch, data_length = dataset.iterator()

            # 알고리즘
            A, B = next_batch
            with tf.variable_scope("shared_variables", reuse=tf.AUTO_REUSE) as scope:

                with tf.name_scope("AtoB_Generator"):
                    AtoB_gene = generator(images=A, name="AtoB_generator")

                with tf.name_scope("BtoA_generator"):
                    BtoA_gene = generator(images=B, name="BtoA_generator")

                #A -> B -> A
                with tf.name_scope("Back_to_A"):
                    BackA = generator(images=AtoB_gene, name="BtoA_generator")

                # B -> A -> B
                with tf.name_scope("Back_to_B"):
                    BackB = generator(images=BtoA_gene, name="AtoB_generator")

                with tf.name_scope("AtoB_Discriminator"):
                    AtoB_Dreal = discriminator(input=B, name="AtoB_Discriminator")
                    # scope.reuse_variables()
                    AtoB_Dgene = discriminator(input=AtoB_gene, name="AtoB_Discriminator")

                with tf.name_scope("BtoA_Discriminator"):
                    BtoA_Dreal = discriminator(input=A, name="BtoA_Discriminator")
                    # scope.reuse_variables()
                    BtoA_Dgene = discriminator(input=BtoA_gene, name="BtoA_Discriminator")

            AtoB_varD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_variables/AtoB_Discriminator')
            # set으로 중복 제거 하고, 다시 list로 바꾼다.

            AtoB_varG = list(set(np.concatenate(
                (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_variables/AtoB_generator'),
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/AtoB_generator')),
                axis=0)))

            BtoA_varD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_variables/BtoA_Discriminator')

            # set으로 중복 제거 하고, 다시 list로 바꾼다.
            BtoA_varG = list(set(np.concatenate(
                (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_variables/BtoA_generator'),
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_variables/BtoA_generator')),
                axis=0)))

            # Adam optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
            with tf.name_scope("saver"):
                saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
                saver_AtoB_generator = tf.train.Saver(var_list=AtoB_varG, max_to_keep=3)
                saver_BtoA_generator = tf.train.Saver(var_list=BtoA_varG, max_to_keep=3)

            '''
            논문에 나와있듯이, log likelihood objective 대신 least-square loss를 사용한다.
            '''
            with tf.name_scope("AtoB_Discriminator_loss"):
                # for AtoB discriminator
                AtoB_DLoss = tf.reduce_mean(tf.square(AtoB_Dreal - tf.ones_like(AtoB_Dreal))) + tf.reduce_mean(
                    tf.square(AtoB_Dgene - tf.zeros_like(AtoB_Dgene)))
            with tf.name_scope("AtoB_Generator_loss"):
                # for AtoB generator
                AtoB_GLoss = tf.reduce_mean(tf.square(AtoB_Dgene - tf.ones_like(AtoB_Dgene)))

            with tf.name_scope("BtoA_Discriminator_loss"):
                # for AtoB discriminator
                BtoA_DLoss = tf.reduce_mean(tf.square(BtoA_Dreal - tf.ones_like(BtoA_Dreal))) + tf.reduce_mean(
                    tf.square(BtoA_Dgene - tf.zeros_like(BtoA_Dgene)))
            with tf.name_scope("BtoA_Generator_loss"):
                # for AtoB generator
                BtoA_GLoss = tf.reduce_mean(tf.square(BtoA_Dgene - tf.ones_like(BtoA_Dgene)))

            # Cycle Consistency Loss
            if cycle_consistency_loss == "L1":
                with tf.name_scope("{}_loss".format(cycle_consistency_loss)):
                    cycle_loss = tf.losses.absolute_difference(A, BackA) + tf.losses.absolute_difference(B, BackB)
                    tf.summary.scalar("{} Loss".format(cycle_consistency_loss), cycle_loss)
                    AtoB_GLoss += tf.multiply(cycle_loss, cycle_consistency_loss_weight)
            else:  # cycle_consistency_loss == "L2"
                with tf.name_scope("{}_loss".format(cycle_consistency_loss)):
                    cycle_loss = tf.losses.mean_squared_error(BackA, A) + tf.losses.mean_squared_error(BackB, B)
                    tf.summary.scalar("{} Loss".format(cycle_consistency_loss), cycle_loss)
                    BtoA_GLoss += tf.multiply(cycle_loss, cycle_consistency_loss_weight)

            with tf.name_scope("AtoB_Discriminator_trainer"):
                AtoB_D_train_op = training(AtoB_DLoss, AtoB_varD, scope='shared_variables/AtoB_Discriminator')
            with tf.name_scope("AtoB_Generator_trainer"):
                AtoB_G_train_op = training(AtoB_GLoss, AtoB_varG, scope='shared_variables/AtoB_generator')
            with tf.name_scope("BtoA_Discriminator_trainer"):
                BtoA_D_train_op = training(BtoA_DLoss, BtoA_varD, scope='shared_variables/BtoA_Discriminator')
            with tf.name_scope("BtoA_Generator_trainer"):
                BtoA_G_train_op = training(BtoA_GLoss, BtoA_varG, scope='shared_variables/BtoA_generator')

            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

            '''
            WHY? 아래 3줄의 코드를 적어 주지 않고, 학습을 하게되면, TEST부분에서 tf.train.import_meta_graph를 사용할 때 오류가 난다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 저장을 해놓은 뒤 TEST 시에 불러와서 다시 사용 해야한다.
            '''
            tf.add_to_collection('A', A)
            tf.add_to_collection('B', B)
            #graph 구조를 파일에 쓴다.
            saver_AtoB_generator.export_meta_graph(os.path.join(model_name, "AtoB.meta"), collection_list=['A', 'B'])
            saver_BtoA_generator.export_meta_graph(os.path.join(model_name, "BtoA.meta"), collection_list=['A', 'B'])

            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            with tf.Session(graph=JG_Graph, config=config) as sess:
                print("initializing!!!")
                sess.run(tf.global_variables_initializer())
                ckpt_all = tf.train.get_checkpoint_state(os.path.join(model_name, 'All'))

                if (ckpt_all and tf.train.checkpoint_exists(ckpt_all.model_checkpoint_path)):
                    print("all variable retored except for optimizer parameter")
                    print("Restore {} checkpoint!!!".format(os.path.basename(ckpt_all.model_checkpoint_path)))
                    saver_all.restore(sess, ckpt_all.model_checkpoint_path)

                summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)
                sess.run(iterator.initializer)
                for epoch in tqdm(range(1, training_epochs + 1)):

                    #논문에서 100epoch가 넘으면 선형적으로 학습률(learning rate)을 감소시킨다고 했다.
                    if epoch > 100:
                        learning_rate*=0.999

                    AtoB_DLoss = 0
                    AtoB_GLoss = 0
                    BtoA_DLoss = 0
                    BtoA_GLoss = 0

                    # 아래의 두 변수가 각각 0.5 씩의 값을 갖는게 가장 이상적이다.
                    AtoB_sigmoidD = 0
                    AtoB_sigmoidG = 0

                    # 아래의 두 변수가 각각 0.5 씩의 값을 갖는게 가장 이상적이다.
                    BtoA_sigmoidD = 0
                    BtoA_sigmoidG = 0

                    total_batch = int(data_length / batch_size)
                    for i in range(total_batch):
                        _, AtoB_D_Loss, AtoB_Dreal_simgoid = sess.run([AtoB_D_train_op, AtoB_DLoss, AtoB_Dreal],
                                                                      feed_dict={lr: learning_rate})
                        _, AtoB_G_Loss, AtoB_Dgene_simgoid = sess.run([AtoB_G_train_op, AtoB_GLoss, AtoB_Dgene],
                                                                      feed_dict={lr: learning_rate})
                        _, BtoA_D_Loss, BtoA_Dreal_simgoid = sess.run([BtoA_D_train_op, BtoA_DLoss, BtoA_Dreal],
                                                                      feed_dict={lr: learning_rate})
                        _, BtoA_G_Loss, BtoA_Dgene_simgoid = sess.run([BtoA_G_train_op, BtoA_GLoss, BtoA_Dgene],
                                                                      feed_dict={lr: learning_rate})

                        AtoB_DLoss += (AtoB_D_Loss / total_batch)
                        AtoB_GLoss += (AtoB_G_Loss / total_batch)
                        BtoA_DLoss += (BtoA_D_Loss / total_batch)
                        BtoA_GLoss += (BtoA_G_Loss / total_batch)

                        AtoB_sigmoidD += AtoB_Dreal_simgoid / total_batch
                        AtoB_sigmoidG += AtoB_Dgene_simgoid / total_batch
                        BtoA_sigmoidD += BtoA_Dreal_simgoid / total_batch
                        BtoA_sigmoidG += BtoA_Dgene_simgoid / total_batch

                        print("{} epoch : {} batch running of {} total batch...".format(epoch, i, total_batch))

                    print("<<< AtoB Discriminator mean output : {} / AtoB Generator mean output : {} >>>".format(
                        np.mean(AtoB_sigmoidD), np.mean(AtoB_sigmoidG)))
                    print("<<< BtoA Discriminator mean output : {} / BtoA Generator mean output : {} >>>".format(
                        np.mean(AtoB_sigmoidD), np.mean(AtoB_sigmoidG)))
                    print("<<< AtoB Discriminator Loss : {} / AtoB Generator Loss  : {} >>>".format(AtoB_DLoss, AtoB_GLoss))
                    print("<<< BtoA Discriminator Loss : {} / BtoA Generator Loss  : {} >>>".format(BtoA_DLoss, BtoA_GLoss))

                    if epoch % display_step == 0:
                        summary_str = sess.run(summary_operation)
                        summary_writer.add_summary(summary_str, global_step=epoch)

                        save_all_model_path = os.path.join(model_name, 'All/')
                        save_AtoB_generator_model_path = os.path.join(model_name, 'AtoB_Generator/')
                        save_BtoA_generator_model_path = os.path.join(model_name, 'BtoA_Generator/')

                        if not os.path.exists(save_all_model_path):
                            os.makedirs(save_all_model_path)
                        if not os.path.exists(save_AtoB_generator_model_path):
                            os.makedirs(save_AtoB_generator_model_path)
                        if not os.path.exists(save_BtoA_generator_model_path):
                            os.makedirs(save_BtoA_generator_model_path)

                        saver_all.save(sess, save_all_model_path, global_step=epoch,
                                       write_meta_graph=False)
                        saver_AtoB_generator.save(sess, save_AtoB_generator_model_path,
                                                  global_step=epoch,
                                                  write_meta_graph=False)
                        saver_BtoA_generator.save(sess, save_BtoA_generator_model_path,
                                                  global_step=epoch,
                                                  write_meta_graph=False)
                print("Optimization Finished!")

    else:
        tf.reset_default_graph()
        AtoB_meta_path=glob.glob(os.path.join(model_name,'*.meta'))
        if len(meta_path)==0:
            print("Lotto Graph가 존재 하지 않습니다.")
            exit(0)
        else:
            print("Lotto Graph가 존재 합니다.")

        # print(tf.get_default_graph()) #기본그래프이다.
        JG = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG.as_default():  # as_default()는 JG를 기본그래프로 설정한다.
            '''
            WHY? 아래 3줄의 코드를 적어 주지 않으면 오류가 난다. 
            -> 단순히 그래프를 가져오고 가중치를 복원하는 것만으로는 안된다. 세션을 실행할때 인수로 사용할 변수에 대한 
            추가 접근을 제공하지 않기 때문에 아래와 같이 get_colltection으로 입,출력 변수들을 불러와서 다시 사용 해야 한다.
            '''
            saver = tf.train.import_meta_graph(meta_path[0], clear_devices=True)  # meta graph 읽어오기
            if saver==None:
                print("meta 파일을 읽을 수 없습니다.")
                exit(0)

            A = tf.get_collection('A')[0]
            B = tf.get_collection('B')[0]
            AtoB_gene = tf.get_collection('AtoB')[0]
            BtoA_gene = tf.get_collection('BtoA')[0]

            # # DB 이름도 추가
            if AtoB:
                model_name = "AtoB_" + model_name
                save_path = "AtoB_" + save_path
            else:
                model_name = "BtoA_" + model_name
                save_path = "BtoA_" + save_path

            if AtoB:
                selection = 0
                gene = AtoB_gene
            else:
                selection = 1
                gene = BtoA_gene

            sess.run(iterator.initializer)
            for i in range(data_length):
                translated_image, batch = sess.run([gene, next_batch])
                visualize(model_name=model_name, named_images=[i, batch[selection], translated_image[0]],
                          save_path=save_path)


if __name__ == "__main__":
    model(TEST=False, AtoB=True, DB_name="maps", use_TFRecord=True, cycle_consistency_loss="L1",
          cycle_consistency_loss_weight=10,
          optimizer_selection="Adam", beta1=0.9, beta2=0.999,  # for Adam optimizer
          decay=0.999, momentum=0.9,  # for RMSProp optimizer
          use_identity_mapping=False,
          norm_selection="instance_norm",  # "instance_norm" or 아무거나
          learning_rate=0.0002, training_epochs=200, batch_size=1, display_step=1,
          save_path="translated_image")  # 학습 완료 후 변환된 이미지가 저장될 폴더 , AtoB=True -> AtoB_가 붙고, False -> BtoA_가 붙는다.
    print("model imported")
