import tensorflow as tf
import glob
import os
import urllib

'''
데이터셋은 아래에서 받는다.
https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz
'''
'''나만의 이미지 데이터셋 만들기 - 텐서플로우의 API 만을 이용하여 만듭니다.
1.  - (데이터셋의 양이 많으면 TFRecord를 쓰는 것도 고려하는 것도 좋다)
2. tf.read_file, tf.random_crop, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리 한다.
3. tf.data.Dataset API를 사용하여 학습이 가능한 데이터 형태로 만든다. 
'''
class Dataset(object):

    def __init__(self):
        pass

    def __repr__(self):
        return "Dataset"

    def image_processing(self, filename, label):
        img_path = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(img_path, channels=3)


    def maybe_download(self):

        """Download a file if not present, and make sure it's the right size."""
        # Step 1: Download the data.
        if not os.path.exists(self.filename):
            self.filename, _ = urllib.request.urlretrieve(self.url + self.filename, self.filename)
        statinfo = os.stat(self.filename)
        if statinfo.st_size == self.expected_bytes:
            print('Found and verified', self.filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + self.filename + '. Can you get to it with a browser?')

    def build_dataset(self):
        pass

    def generate_batch(self):
        pass

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
