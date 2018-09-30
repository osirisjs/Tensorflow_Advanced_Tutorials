import os

import gym
import numpy as np
import tensorflow as tf
from tqdm import *

# 핸즈온 머신러닝 책의 Cartpole 예제 참고 및 수정
'''
정책 파라미터에 대한 보상의 그라디언트를 평가해서 높은 보상의 방향을 따르는
그라디언트로('경사 상승법') 파라미터를 수정하는 최적화 기법을 사용 - Policy Gradient(정책 그래디언트)

PG 알고리즘 : 높은 보상을 얻는 방향의 그라디언트로 정책의 파라미터를 최적화하는 알고리즘
 - 로날드 윌리엄스의 REINFOCE 알고리즘이 유명함
 - 일단 에피소드(게임)를 몇 번 진행해보고 이를 평균 내어 학습하기 때문에 몬테카를로 정책 그래디언트(Monte Carlo Policy Gradient)라고도 함
'''


class CartPole(object):

    def __init__(self, Train=True, epoch=1000, game_step=1000, gradient_update=10, learning_rate=0.01,
                 discount_factor=0.95, save_weight=100, save_path="CartPole", rendering=False):

        self.Train = Train

        self.env = gym.make("CartPole-v0")
        self.n_hidden = 4
        self.n_input = 4
        self.n_output = 1

        self.epoch = epoch  # 학습 횟수
        self.initializer = tf.contrib.layers.variance_scaling_initializer()  # He 초기화
        self.learning_rate = learning_rate  # 학습률
        self.game_step = game_step  # 게임별 최대 스텝
        self.gradient_update = gradient_update  # 10번의 게임이 끝난 후 정책을 훈련한다
        self.save_weight = save_weight  # 10번의 게임이 끝날때마다 모델을 저장한다.
        self.discount_factor = discount_factor  # 할인 계수
        self.rendering = rendering  # 학습 시 애니메이션을 볼지 말지 : 안봐야 학습이 빠르다.
        self.save_path = save_path  # 가중치가 저장될 경로

    # 행동 평가 : 신용 할당 문제 -> 할인 계수 도입
    def discount_rewards(self, rewards, discount_factor):
        discount_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_factor
            discount_rewards[step] = cumulative_rewards
        return discount_rewards

    # 행동 평가 : 신용 할당 문제 -> 할인 계수 도입
    def disconut_and_normalize_rewars(self, all_rewards, discount_factor):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_factor) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards, axis=0)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def Graph(self):

        JG_Graph = tf.Graph()  # 내 그래프로 설정한다.- 혹시라도 나중에 여러 그래프를 사용할 경우를 대비
        with JG_Graph.as_default():  # as_default()는 JG_Graph를 기본그래프로 설정한다.

            x = tf.placeholder(tf.float32, shape=(None, self.n_input))
            hidden = tf.layers.dense(x, self.n_hidden, activation=tf.nn.elu, kernel_initializer=self.initializer)
            logits = tf.layers.dense(hidden, self.n_output, kernel_initializer=self.initializer)
            outputs = tf.nn.sigmoid(logits)

            '''
            행동 선택
            output = 1 -> action : 0
            output = 0 -> action : 1
            '''
            p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
            action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
            saver = tf.train.Saver()

            if self.Train:
                # 라벨값
                y = 1. - tf.to_float(action)
                '''
                log(p)를 커지는 방향으로 그라디언트를 업데이트 해야하므로
                sigmoid_cross_entropy 를 최소화 하는 것과 같다.
                '''
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                grads_and_vars = optimizer.compute_gradients(cross_entropy)

                # <<< credit assignment problem(신용 할당 문제) 을 위한 작업 >>> #
                gradients = [grad for grad, variable in grads_and_vars]
                gradient_placeholders = []
                grads_and_vars_feed = []

                for grad, variable in grads_and_vars:
                    # credit assignment problem(신용 할당 문제)를 위해 담아둘공간이 필요해서 아래와 같은 작업 진행
                    gradient_placeholder = tf.placeholder(tf.float32, shape=None)
                    gradient_placeholders.append(gradient_placeholder)

                    grads_and_vars_feed.append((gradient_placeholder, variable))
                training_op = optimizer.apply_gradients(grads_and_vars_feed)

            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.1

            with tf.Session(graph=JG_Graph, config=config) as sess:
                sess.run(tf.global_variables_initializer())
                ckpt = tf.train.get_checkpoint_state(self.save_path)
                if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                    print("<<< all variable retored except for optimizer parameter >>>")
                    print("<<< Restore {} checkpoint!!! >>>".format(os.path.basename(ckpt.model_checkpoint_path)))
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print("<<< initializing!!! >>>")

                if self.Train:
                    for epoch in tqdm(range(1, self.epoch + 1, 1)):
                        all_rewards = []
                        all_gradients = []
                        for game in range(self.gradient_update):
                            current_rewards = []
                            current_gradients = []
                            obs = self.env.reset()
                            for step in range(self.game_step):
                                if self.rendering:
                                    self.env.render()  # 학습 속도를 빠르게 하려면 요부분을 주석 처리하라.
                                # action = tf.multinomial(tf.log(p_left_and_right), num_samples=1) 가 2차원 배열로 반환!!!
                                action_val, gradients_val = sess.run([action, gradients],
                                                                     feed_dict={x: obs.reshape(1, self.n_input)})
                                obs, reward, done, info = self.env.step(action_val[0][0])

                                # 모든 행동을 다 고려하다니.. 오래 걸릴 수 밖에 없구나...
                                current_rewards.append(reward)
                                current_gradients.append(gradients_val)
                                if done:
                                    break
                            # self.update_periods(ex) 10 게임) 마다 보상과 그라디언트를 append 한다.
                            all_rewards.append(current_rewards)
                            all_gradients.append(current_gradients)

                        ''' 
                        정규화된 결과가 나온다. 왜 정규화를 해야하나?
                        책에는? -> 행동에 대해 신뢰할만한 점수를 얻으려면 많은 에피소드(게임)를 실행하고
                        모든 행동의 점수를 정규화 해야한다고 나와있다.
                        '''
                        all_rewards = self.disconut_and_normalize_rewars(all_rewards, self.discount_factor)
                        feed_dict = {}

                        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
                            # 모든 에피소드와 모든 스텝에 걸쳐 그라디언트와 보상점수를 곱한다. -> 이후 각 그라디언트에 대해 평균을 구한다.
                            # 이래서 오래 걸린다...
                            mean_gradients = np.mean(
                                [reward * all_gradients[game_index][step][var_index] for game_index, rewards in
                                 enumerate(all_rewards) for step, reward in enumerate(rewards)], axis=0)

                            feed_dict[gradient_placeholder] = mean_gradients
                        # 평균 그라디언트를 훈련되는 변수마다 하나씩 주입하여 훈련연산을 실행한다.
                        sess.run(training_op, feed_dict=feed_dict)
                        if epoch % self.save_weight == 0:
                            saver.save(sess, self.save_path + "/Cartpole.ckpt", global_step=epoch)

                else:
                    obs = self.env.reset()
                    for _ in tqdm(range(self.game_step)):
                        self.env.render()
                        action_val = sess.run(action, feed_dict={x: obs.reshape(1, self.n_input)})
                        obs, reward, done, info = self.env.step(action_val[0][0])
                        if done:
                            break

                # Rendering 문제 생길 시 아래 주석을 풀어보기
                # try:
                #     del self.env
                # except ImportError:
                #     pass


if __name__ == "__main__":
    CartPole(Train=True, epoch=1, game_step=1000, gradient_update=10, learning_rate=0.01,
             discount_factor=0.95, save_weight=100, save_path="CartPole", rendering=True).Graph()
