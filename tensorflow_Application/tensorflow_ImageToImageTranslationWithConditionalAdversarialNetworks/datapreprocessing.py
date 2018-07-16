import tensorflow as tf

class Dataset(object):
    def __init__(self):
        pass

    def __repr__(self):
        return "Dataset"

    def DataLoader(batch_size=None):
        # 1.데이터셋 읽기
        data = pd.read_excel("lotto.xlsx")
        data = np.asarray(data)
        input = data[1:, 1:]
        output = data[0:np.shape(data)[0] - 1, 1:]
        data = np.flipud(input).astype(np.float32)
        label = np.flip(output, axis=0).astype(np.float32)

        # Tensorflow 데이터셋 만들기 -
        '''tensorflow의 tf.data.Dataset utility를 사용했습니다. 
        직접 batch, shuffle등의 코드를 구현할 필요가 없습니다.!!! 
        '''
        dataset = tf.data.Dataset.from_tensor_slices((data.reshape((-1, 6)), label))  # 데이터셋 가져오기
        dataset = dataset.shuffle(100000).repeat().batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next(), len(data) + 1
