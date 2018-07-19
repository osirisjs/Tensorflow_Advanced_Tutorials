import os
import tarfile
import urllib.request

import tensorflow as tf

'''
데이터셋은 아래에서 받았다.
https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz
'''
'''나만의 이미지 데이터셋 만들기 - 텐서플로우의 API 만을 이용하여 만들자.
많은 것을 지원해주는 텐서플로우 API 만을 이용해서 만들 경우 코드가 굉장히 짧아지면서 빠르다.
그러나, 공부해야할 것이 꽤 많다. tf.data.Dataset API, TFRecord, tf.image API 등등 여러가지를 알아야 한다. 

1. 데이터를 다운로드한다. 데이터가 대용량이므로 텐서플로의 기본 데이터 형식인 TFRecord(프로토콜버퍼, 직렬화) 형태로 바꾼다.
    -> 텐서플로로 바로 읽어 들일 수 있는 형식, 입력 파일들을 하나의 통합된 형식으로 변환하는 것(하나의 덩어리)
    -> 정리하자면 Tensor 인데, 하나의 덩어리 형태
2. tf.data.Dataset API를 사용하여 학습이 가능한 데이터 형태로 만든다. 
    -> tf.read_file, tf.random_crop, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리 한다.
'''


class Dataset(object):

    def __init__(self, url="", batch_size=4):

        self.url = url

        self.Dataset_Path = "Dataset"
        if not os.path.exists(self.Dataset_Path):
            os.makedirs(self.Dataset_Path)

        self.dataset_name = os.path.join(self.Dataset_Path, "edges2shoes")
        self.file_name = self.dataset_name + ".tar.gz"
        self.batch_size = batch_size
        self.Preparing_Learning_Dataset()  # 데이터셋 다운로드
        # self.Generate_Batch() # 학습형태에 맞게 다운로드

    def __repr__(self):
        return "Dataset Preprocessing"

    def image_preprocessing(self):
        pass

    def Preparing_Learning_Dataset(self):

        # 1. 데이터셋 폴더가 존재하지 않으면 다운로드(다운로드 시간이 굉장히 오래걸립니다.)
        if not os.path.exists(self.dataset_name):  # 데이터셋 폴더가 존재하지 않는 다면?
            if not os.path.exists(self.file_name):  # 데이터셋 압축 파일이 존재하지 않는 다면, 다운로드
                self.file_name, _ = urllib.request.urlretrieve(self.url, self.file_name)
                print("{} Download Completed".format(self.file_name))
            else:  # 만약 데이터셋 압축 파일이 존재한다면, 존재한다고 print를 띄워주자.
                print("{} Exists".format(self.file_name))
            # 2. 압축파일이 있는 상태이므로 압축을 푼다
            with tarfile.open(self.file_name) as tf:
                tf.extractall()
            print("{} Unzip Completed".format(self.file_name))
            # 3. 용량차지 하므로, tar.gz 파일을 지워주자. -> 데이터셋 폴더가 존재하므로
            # os.remove(self.file_name) # 하드디스크에 용량이 충분하다면, 굳이 필요 없는 코드다.
        else:  # 데이터셋 폴더가 존재한다면, 존재한다고 print를 띄워주자
            print("edges2shoes Dataset Exists")

        '''3. 데이터형식을 텐서플로의 기본 데이터 형식인 TFRecord 로 바꾼다.(대용량의 데이터를 처리하므로 TFRecord를 사용하는게 좋다.)
        # 바꾸기전에 데이터를 입력, 출력으로 나눈 다음 덩어리로 저장한다. -> Generate_Batch의 map함수에서 입력, 출력 값으로 분리해도 되지만
        이런 전처리는 미리 되있어야 한다.
        '''

    def Generate_Batch(self):

        # 4. TFRecordDataset()사용해서 읽어오기
        file_name = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(file_name)
        dataset = dataset.map()  #
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=self.batch_size)
        iterator = dataset.make_initializable_iterator()
        training_file_name = ["", ""]
        return iterator

    '''
    Random jitter was applied by resizing th0e 256256 input
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


if __name__ == "__main__":

    url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz"
    Dataset(url=url, batch_size=4)

else:
    print("model imported")
