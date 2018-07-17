import tensorflow as tf

'''나만의 이미지 데이터셋 만들기 - 텐서플로우의 API만 사용합니다.

1. 우선 이미지의 경로를 불러온다.
2. 메모리를 일정한 변수에 저장을 해놓은다. - (데이터셋의 양이 많으면 TFRecord를 쓰는 것도 고려하는 것도 좋다)
3. tf.read_file, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리 한다.
4. tf.data.Dataset API를 사용하여 학습이 가능한 데이터 형태로 만든다. - 끝
'''
class Dataset(object):

    def __init__(self):
        pass

    def __repr__(self):
        return "Dataset"

    '''
    Random jitter was applied by resizing the 256256 input
    images to 286 x 286 and then randomly cropping back to
    size 256 x 256
    # '''
    # def DataLoader(batch_size=None):
    #     # 1.데이터셋 읽기
    #
    #     # Tensorflow 데이터셋 만들기 -
    #     '''tensorflow의 tf.data.Dataset utility를 사용했습니다.
    #     직접 batch, shuffle등의 코드를 구현할 필요가 없습니다.!!!
    #     '''
    #     dataset = tf.data.Dataset.from_tensor_slices((data.reshape((-1, 6)), label))  # 데이터셋 가져오기
    #     dataset = dataset.shuffle(100000).repeat().batch(batch_size)
    #     iterator = dataset.make_one_shot_iterator()
    #     return iterator.get_next(), len(data) + 1
