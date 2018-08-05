import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from tqdm import tqdm

from data_preprocessing import data_preprocessing


def Word2Vec(TEST=True, tSNE=True, model_name="Word2Vec", weight_selection="encoder",  # encoder or decoder
         vocabulary_size=30000, tSNE_plot=500, similarity_number=8,
         # similarity_number -> 비슷한 문자 출력 개수
         # num_skip : 하나의 문장당 num_skips 개의 데이터를 생성
         validation_number=30, embedding_size=192, batch_size=192, num_skips=8, window_size=4,
         negative_sampling=64, optimizer_selection="SGD", learning_rate=0.1, training_epochs=100,
         display_step=1, weight_sharing=False, *Arg, **kwargs):

    if weight_sharing:
        model_name="WeigthSharing_"+ model_name

    dp = data_preprocessing(url='http://mattmahoney.net/dc/',
                            filename='text8.zip',
                            expected_bytes=31344016,
                            vocabulary_size=vocabulary_size)

    if TEST == False:
        if os.path.exists("tensorboard/{}".format(model_name)):
            shutil.rmtree("tensorboard/{}".format(model_name))

    # encoder, decoder의 가중치중 어떤 것을 쓸지 결정해야함
    def evaluate(embedding_matrix=None, validation_inputs=None):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_matrix), axis=1, keepdims=True))
        normalized = tf.divide(embedding_matrix, norm)
        val_embeddings = tf.nn.embedding_lookup(normalized, validation_inputs)
        '''  
        --> embedding_matrix 가 잘 학습이 되었다면, 비슷한 단어들은 비슷한 벡터의 값을 가질 것이다. - TSNE로 확인 가능하다.
    
        이 embedding_matrix 을 
        decoder로 출력했을 때, 상위 몇개의 단어는 연관된 의미를 표현할 것이다. 
        - cosine_similarity 은 결국은 decoder의 출력이다 -> shape = (validation_number, vocabulary_size)
        -> 여기서 가장 큰 값이 입력과 같아야 하는 것이고, 최상위 몇개의 값이 가장 큰값(=입력)과 연관이 있는
        단어들을 의미한다고 한다. -> 정말 신기한것 같다.(물론 데이터가 좋아야 한다.)
        '''
        cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b=True)
        return normalized, cosine_similarity

    def embedding_layer(embedding_shape=None, train_inputs=None):
        with tf.variable_scope("embedding"):
            embedding_init = tf.random_uniform(embedding_shape, minval=-1, maxval=1)
            embedding_matrix = tf.get_variable("E", initializer=embedding_init)
            return tf.nn.embedding_lookup(embedding_matrix, train_inputs), embedding_matrix

    def noise_contrastive_loss(weight_shape=None, bias_shape=None, train_labels=None, embed=None, num_sampled=None,
                               num_classes=None, encoder_weight=None, weight_sharing=True):

        nce_weight_init = tf.truncated_normal(weight_shape, stddev=np.sqrt(1.0 / (weight_shape[1])))
        nce_bias_init = tf.zeros(bias_shape)
        nce_w = tf.get_variable("w", initializer=nce_weight_init)
        nce_b = tf.get_variable("b", initializer=nce_bias_init)

        if weight_sharing:
            total_loss = tf.nn.nce_loss(weights=encoder_weight, biases=nce_b, labels=train_labels, inputs=embed, \
                                        num_sampled=num_sampled, num_classes=num_classes)
            nce_w=encoder_weight
        else:
            with tf.variable_scope("nce", reuse=tf.AUTO_REUSE) as scope:
                total_loss = tf.nn.nce_loss(weights=nce_w, biases=nce_b, labels=train_labels, inputs=embed, \
                                                num_sampled=num_sampled, num_classes=num_classes)

        return tf.reduce_mean(total_loss), nce_w

    def training(cost, global_step):
        tf.summary.scalar("cost", cost)
        if optimizer_selection == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_selection == "RMSP":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_selection == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_operation = optimizer.minimize(cost, global_step=global_step)
        return train_operation

    # print(tf.get_default_graph()) #기본그래프이다.
    JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
    with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.
        with tf.name_scope("feed_dict"):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, [batch_size, 1])

        with tf.variable_scope("Skip_Gram", reuse=tf.AUTO_REUSE) as scope:
            embed, e_matrix = embedding_layer(embedding_shape=(dp.vocabulary_size, embedding_size),
                                              train_inputs=train_inputs)
            # scope.reuse_variables()


        with tf.name_scope("nce_loss"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            cost, nce_weight = noise_contrastive_loss(weight_shape=(dp.vocabulary_size, embedding_size), \
                                                      bias_shape=(dp.vocabulary_size), \
                                                      train_labels=train_labels, \
                                                      embed=embed, \
                                                      num_sampled=negative_sampling, \
                                                      num_classes=dp.vocabulary_size, \
                                                      encoder_weight=e_matrix, \
                                                      weight_sharing=weight_sharing)

        # Adam optimizer의 매개변수들을 저장하고 싶지 않다면 여기에 선언해야한다.
        with tf.name_scope("saver"):
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)

        if not TEST:
            with tf.name_scope("trainer"):
                train_operation = training(cost, global_step)
            with tf.name_scope("tensorboard"):
                summary_operation = tf.summary.merge_all()

        # 학습이 아주 잘될 경우 아래의 두 결과가 다 좋아야한다.
        if TEST:
            # validation_inputs = tf.constant(np.random.choice(dp.vocabulary_size, validation_number, replace=False), dtype=tf.int32)
            # Data preprocessing 에서 가장 많이 출현한 10000개의 단어를 학습데이터로 사용
            # 학습데이터중 앞에서부터 tSNE_plot개의 단어중 validation_numbe개를 무작위로 선택하여 비슷한 단어들을 출력하는 변수.
            validation_inputs = tf.constant(np.random.choice(tSNE_plot, validation_number, replace=False),
                                            dtype=tf.int32)

            if weight_selection == "encoder":
                using_encoder_evaluate_operation = evaluate(embedding_matrix=e_matrix,
                                                            validation_inputs=validation_inputs)
            else:
                using_decoder_evaluate_operation = evaluate(embedding_matrix=nce_weight,
                                                            validation_inputs=validation_inputs)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=JG_Graph, config=config) as sess:
        print("initializing!!!")
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join('model',model_name))

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Restore {} checkpoint!!!".format(os.path.basename(ckpt.model_checkpoint_path)))
            saver.restore(sess, ckpt.model_checkpoint_path)

        if not TEST:
            batches_per_epoch = int(
                (dp.vocabulary_size * num_skips) / batch_size)  # Number of batches per epoch of training
            summary_writer = tf.summary.FileWriter(os.path.join("tensorboard", model_name), sess.graph)
            for epoch in tqdm(range(training_epochs)):
                avg_cost = 0.
                for minibatch in range(batches_per_epoch):  # # Number of batches per epoch of training
                    mbatch_x, mbatch_y = dp.generate_batch(batch_size=batch_size, num_skips=num_skips,
                                                           window_size=window_size)
                    feed_dict = {train_inputs: mbatch_x, train_labels: mbatch_y}
                    _, new_cost = sess.run([train_operation, cost], feed_dict=feed_dict)
                    # Compute average loss
                    avg_cost += new_cost / batches_per_epoch
                print("cost : {0:0.3}".format(avg_cost))

                if epoch % display_step == 0:
                    summary_str = sess.run(summary_operation, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, global_step=sess.run(global_step))
                    save_path=os.path.join('model', model_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    saver.save(sess, save_path + "/", global_step=sess.run(global_step),
                               write_meta_graph=False)
            print("Optimization Finished!")

        if tSNE and TEST:
            # 비슷한 단어 출력해보기
            if weight_selection == "encoder":
                final_embeddings, similarity = sess.run(using_encoder_evaluate_operation)
            else:
                final_embeddings, similarity = sess.run(using_decoder_evaluate_operation)

            for i in range(validation_number):
                val_word = dp.reverse_dictionary[sess.run(validation_inputs)[i]]

                '''argsort()함수는 오름차순으로 similarity(batch_size, vocabulary size) 값을
                정렬한다. 즉, 가장 높은 확률을 가진 놈은 가장 오른쪽에 있다. 따라서 뒤에서부터 접근하여
                큰 값들을 가져온다.
                '''
                neighbors = (similarity[i, :]).argsort()[-1:-1-similarity_number-1:-1]  # -1 ~ -similarity_number-1 까지의 값
                # neighbors = (-similarity[i, :]).argsort()[0:similarity_number+1]  # -1 ~ -similarity_number-1 까지의 값
                print_str = "< Nearest neighbor of {} / ".format(val_word)

                # word2vec도 결국은 autoencoder 이다. 가장 가까운 단어는 자기 자신이 나온다.[index = 0]
                # 해당 단어와 가장 가까운 것은 자기 자신이기 때문에 학습 하는 과정에서 자연스럽게 정해진다.
                '''
                학습이 전혀 되지 않는 상태에서, Test만 해도 0번은 자기 자신이 나오는데 이는
                embedding_init가 균일분포로 초기화 되었기 때문이다.(tf.random_uniform(embedding_shape, minval=-1, maxval=1))
                embedding_init가 1이나 0으로 초기화 되면 이상한 값이 나온다.
                '''
                print_str += "target word : {} > :".format(dp.reverse_dictionary[neighbors[0]])

                for k in range(1, similarity_number + 1, 1):
                    print_str += " {},".format(dp.reverse_dictionary[neighbors[k]])
                print(print_str[:-1])

            # T-SNE로 그려보기
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs = tsne.fit_transform(final_embeddings[:tSNE_plot, :])
            labels = [dp.reverse_dictionary[i] for i in range(tSNE_plot)]

            figure = plt.figure(figsize=(18, 18))  # in inches
            figure.suptitle("Visualizing Word2Vector using T-SNE")
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
            figure.savefig("Word2Vec_Using_TSNE.png", dpi=300)
            plt.show()


if __name__ == "__main__":
    # optimizers_ selection = "Adam" or "RMSP" or "SGD"
    # weight_selection 은 encoder, decoder 임베디중 어떤것을 사용할 것인지 선택하는 변수
    # weight_sharing=True시 weight_selection="decoder"라고 설정해도 weight_selection="encoder"로 강제 설정된다.
    Word2Vec(TEST=True, tSNE=True, model_name="Word2Vec", weight_selection="encoder",  # encoder or decoder
         vocabulary_size=30000, tSNE_plot=500, similarity_number=8,
         # similarity_number -> 비슷한 문자 출력 개수
         # num_skip : 하나의 문장당 num_skips 개의 데이터를 생성
         validation_number=30, embedding_size=192, batch_size=192, num_skips=8, window_size=4,
         negative_sampling=64, optimizer_selection="SGD", learning_rate=0.1, training_epochs=100,
         display_step=1, weight_sharing=False)
else:
    print("word2vec imported")
