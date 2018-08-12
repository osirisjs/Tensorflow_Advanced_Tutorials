import glob
import os
import random
import tarfile
import urllib.request

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

'''
데이터셋은 아래에서 받았다.
https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz
이미지(원본 이미지를)를 분할(Segmentation) 해보자

나만의 이미지 데이터셋 만들기 - 텐서플로우의 API 만을 이용하여 만들자.
많은 것을 지원해주는 텐서플로우 API 만을 이용해서 만들 경우 코드가 굉장히 짧아지면서 빠르다.
하지만, 공부해야할 것이 꽤 많다. TFRecord, tf.data.Dataset API, tf.image API 등등 여러가지를 알아야 한다.

##################내가 생각하는 데이터 전처리기를 만드는 총 4가지 방법################# 
#총 4가지 방법이 있는 것 같다. 
1. numpy로 만들어서 feed_dict하는 방법 - feed_dict 자체가 파이런 런타임에서 텐서플로 런타임에서 데이터를 단일 스레드로 복사해서
이로인해 지연이 발생하고 속도가 느려진다.

2. tf.data.Dataset.from_tensor_slices 로 만든 다음에 그래프에 올려버리는 방법 - 아래의 첫번째 방법 - 현재 코드는 jpg, jpeg 데이터에만 가능
- 약간 어중간한 위치의 데이터 전처리 방법이다. 하지만, 1번보다는 빠르다

3. tf.data.TFRecordDataset를 사용하여 TRRecord(이진 파일, 직렬화된 입력 데이터)라는 텐서플로우 표준 파일형식으로 저장된 파일을 불러온 다음
  그래프에 올려버리는 방법 - 아래의 두번째 방법
  
4. 멀티스레드 사용방법 - 이것은 공부가 더 필요하다. - 추 후 cycleGAN 을 구현 할시 공부해서 구현 해보자 

<첫번째 방법>은 원래의 데이터파일을 불러와서 학습 한다.
<두번째 방법>은 TFRecord(텐서플로우의 표준 파일 형식)으로 원래의 데이터를 저장한뒤 불러와서 학습하는 방식이다.
<두번째 방법>이 빠르다.
<두번쨰 방법>은 모든데이터는 메모리의 하나의 블록에 저장되므로, 입력 파일이 개별로 저장된 <첫번째 방법>에 비헤
메모리에서 데이터를 읽는데 필요한 시간이 단축 된다.

구체적으로, 
<첫번째 방법>
1. 데이터를 다운로드한다. 데이터셋이 들어있는 파일명을 들고와 tf.data.Dataset.from_tensor_slices 로 읽어들인다.
2. tf.data.Dataset API 및 여러가지 유용한 텐서플로우 API 를 사용하여 학습이 가능한 데이터 형태로 만든다. 
    -> tf.read_file, tf.random_crop, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리하고 학습가능한 형태로 만든다.

<두번째 방법> - 메모리에서 데이터를 읽는 데 필요한 시간이 단축된다.
1. 데이터를 다운로드한다. 데이터가 대용량이므로 텐서플로의 기본 데이터 형식인 TFRecord(프로토콜버퍼, 직렬화) 형태로 바꾼다.
    -> 텐서플로로 바로 읽어 들일 수 있는 형식, 입력 파일들을 하나의 통합된 형식으로 변환하는 것(하나의 덩어리)
    -> 정리하자면 Tensor 인데, 하나의 덩어리 형태
2. tf.data.Dataset API 및 여러가지 유용한 텐서플로우 API 를 사용하여 학습이 가능한 데이터 형태로 만든다. 
    -> tf.read_file, tf.random_crop, tf.image.~ API를 사용하여 논문에서 설명한대로 이미지를 전처리하고 학습가능한 형태로 만든다.
'''


class Dataset(object):

    def __init__(self, DB_name="facades", AtoB=False, batch_size=1, use_TrainDataset=True, inference_size=(256, 256)):

        self.Dataset_Path = "Dataset"
        self.DB_name = DB_name
        self.AtoB = AtoB
        self.inference_size = inference_size
        # 학습용 데이터인지 테스트용 데이터인지 알려주는 변수
        self.use_TrainDataset = use_TrainDataset

        # infernece_size의 최소 크기를 (256, 256)로 지정
        if not self.use_TrainDataset:
            if self.inference_size == None and self.inference_size[0] < 256 and self.inference_size[1] < 256:
                print("inference size는 (256,256)보다는 커야 합니다.")
                exit(0)
            else:
                self.height_size = inference_size[0]
                self.width_size = inference_size[1]

        # "{self.DB_name}.tar.gz"의 파일의 크기는 미리 구해놓음. (미리 확인이 필요함.)
        if DB_name == "cityscapes":
            self.file_size = 103441232
        elif DB_name == "facades":
            self.file_size = 30168306
        elif DB_name == "maps":
            self.file_size = 250242400
        else:
            print("Please enter ""DB_name"" correctly")
            print("The program is forcibly terminated.")
            exit(0)

        if not os.path.exists(self.Dataset_Path):
            os.makedirs(self.Dataset_Path)

        self.url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz".format(self.DB_name)
        self.dataset_folder = os.path.join(self.Dataset_Path, self.DB_name)
        self.dataset_targz = self.dataset_folder + ".tar.gz"

        # Test Dataset은 무조건 하나씩 처리하자.
        if self.use_TrainDataset == False:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        # 데이터셋 다운로드 한다.
        self.Preparing_Learning_Dataset()

        if self.use_TrainDataset:

            # TFRecord
            self.file_path_list = glob.glob(os.path.join(self.dataset_folder, "train/*"))
            self.TFRecord_train_path = os.path.join(self.dataset_folder, "TFRecord_train")

            if not os.path.exists(self.TFRecord_train_path):
                os.makedirs(self.TFRecord_train_path)

            if self.AtoB:
                self.TFRecord_path = os.path.join(self.TFRecord_train_path,
                                                  'AtoBtrain.tfrecords')
                self.TFRecord_path = os.path.join(self.TFRecord_train_path,
                                                  'BtoAtrain.tfrecords')
            # TFRecord 파일로 쓰기.
            self.TFRecordWriter()

        else:
            self.file_path_list = glob.glob(os.path.join(self.dataset_folder, "val/*"))
            self.TFRecord_val_path = os.path.join(self.dataset_folder, "TFRecord_val")

            if not os.path.exists(self.TFRecord_val_path):
                os.makedirs(self.TFRecord_val_path)

            if self.AtoB:
                self.TFRecord_path = os.path.join(self.TFRecord_val_path,
                                                  'AtoBval{}x{}.tfrecords'.format(self.height_size, self.width_size))
            else:
                self.TFRecord_path = os.path.join(self.TFRecord_val_path,
                                                  'BtoAval{}x{}.tfrecords'.format(self.height_size,
                                                                                  self.width_size))
            # TFRecord 파일로 쓰기.
            self.TFRecordWriter()

    def __repr__(self):
        return "Dataset Loader"

    def iterator(self):

        iterator, next_batch, db_length = self.Using_TFRecordDataset()

        return iterator, next_batch, db_length

    def Preparing_Learning_Dataset(self):

        # 1. 데이터셋 폴더가 존재하지 않으면 다운로드
        if not os.path.exists(self.dataset_folder):
            if not os.path.exists(self.dataset_targz):  # 데이터셋 압축 파일이 존재하지 않는 다면, 다운로드
                print("<<< {} Dataset Download required >>>".format(self.DB_name))
                urllib.request.urlretrieve(self.url, self.dataset_targz)
                print("<<< {} Dataset Download Completed >>>".format(self.DB_name))

            # "{self.DB_name}.tar.gz"의 파일의 크기는 미리 구해놓음. (미리 확인이 필요함.)
            elif os.path.exists(self.dataset_targz) and os.path.getsize(
                    self.dataset_targz) == self.file_size:  # 완전한 데이터셋 압축 파일이 존재한다면, 존재한다고 print를 띄워주자.
                print("<<< ALL {} Dataset Exists >>>".format(self.DB_name))

            else:  # 데이터셋 압축파일이 존재하긴 하는데, 제대로 다운로드 되지 않은 상태라면, 삭제하고 다시 다운로드
                print(
                    "<<< {} Dataset size must be : {}, but now size is {} >>>".format(self.DB_name, self.file_size,
                                                                                      os.path.getsize(
                                                                                          self.dataset_targz)))
                os.remove(self.dataset_targz)  # 완전하게 다운로드 되지 않은 기존의 데이터셋 압축 파일을 삭제
                print("<<< Deleting incomplete {} Dataset Completed >>>".format(self.DB_name))
                print("<<< we need to download {} Dataset again >>>".format(self.DB_name))
                urllib.request.urlretrieve(self.url, self.dataset_targz)
                print("<<< {} Dataset Download Completed >>>".format(self.DB_name))

            # 2. 완전한 압축파일이 다운로드 된 상태이므로 압축을 푼다
            with tarfile.open(self.dataset_targz) as tar:
                tar.extractall(path=self.Dataset_Path)
            print("<<< {} Unzip Completed >>>".format(os.path.basename(self.dataset_targz)))
            print("<<< {} Dataset now exists >>>".format(self.DB_name))
        else:
            print("<<< {} Dataset is already Exists >>>".format(self.DB_name))

    def _image_preprocessing(self, image):

        # 1. 이미지를 읽는다.
        feature = {'image_left': tf.FixedLenFeature([], tf.string),
                   'image_right': tf.FixedLenFeature([], tf.string),
                   'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64)}

        parser = tf.parse_single_example(image, features=feature)
        img_decoded_raw_left = tf.decode_raw(parser['image_left'], tf.float32)
        img_decoded_raw_right = tf.decode_raw(parser['image_right'], tf.float32)
        height = tf.cast(parser['height'], tf.int32)
        width = tf.cast(parser['width'], tf.int32)

        iL = tf.reshape(img_decoded_raw_left, (height, width, 3))
        iR = tf.reshape(img_decoded_raw_right, (height, width, 3))

        # 4. gerator의 활성화 함수가 tanh이므로, 스케일을 맞춰준다.
        iL_scaled = tf.subtract(tf.divide(iL, 127.5), 1.0)  # gerator의 활성화 함수가 tanh이므로, 스케일을 맞춰준다.
        iR_scaled = tf.subtract(tf.divide(iR, 127.5), 1.0)  # gerator의 활성화 함수가 tanh이므로, 스케일을 맞춰준다.

        input = iL_scaled
        label = iR_scaled

        '''
        논문에서...
        Random jitter was applied by resizing the 256 x 256 input images to 286 x 286
        and then randomly cropping back to size 256 x 256
        '''
        # Train Dataset 에서만 동작하게 하기 위함
        if self.use_TrainDataset:
            # 5. 286x286으로 키운다. - 30씩 더한다
            iL_resized = tf.image.resize_images(images=iL_scaled, size=(tf.shape(iL_scaled)[0]+30, tf.shape(iL_scaled)[1]+30))
            iR_resized = tf.image.resize_images(images=iR_scaled, size=(tf.shape(iR_scaled)[0]+30, tf.shape(iR_scaled)[1]+30))

            # 6. 이미지를 256x256으로 랜덤으로 자른다. - 30
            iL_random_crop = tf.random_crop(iL_resized, size=(tf.shape(iL_resized)[0],tf.shape(iL_resized)[1],tf.shape(iL_resized)[2]))
            iR_random_crop = tf.random_crop(iR_resized, size=(tf.shape(iR_resized)[0],tf.shape(iR_resized)[1],tf.shape(iR_resized)[2]))

            input = iL_random_crop
            label = iR_random_crop

        if self.AtoB:
            return input, label
        else:
            return label, input

    # TFRecord를 만들기위해 이미지를 불러올때 쓴다.
    def load_image(self, address):
        img = cv2.imread(address)

        # RGB로 바꾸기
        if not self.use_TrainDataset:
            img = cv2.resize(img, (self.width_size, self.height_size), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        middle_point = int(img.shape[1] / 2)
        img_left = img[:, :middle_point, :]
        img_right = img[:, middle_point:, :]
        return img_left, img_right

    def TFRecordWriter(self):

        '''데이터형식을 텐서플로의 기본 데이터 형식인 TFRecord 로 바꾼다.(대용량의 데이터를 처리하므로 TFRecord를 사용하는게 좋다.)
        # 바꾸기전에 데이터를 입력, 출력으로 나눈 다음 덩어리로 저장한다. -> Generate_Batch의 map함수에서 입력, 출력 값으로 분리해도 되지만
        이런 전처리는 미리 되있어야 한다.'''
        # http: // machinelearninguru.com / deep_learning / data_preparation / tfrecord / tfrecord.html 참고했다.
        # TFRecord로 바꾸기
        print("<<< Using TFRecord format >>>")
        if not os.path.isfile(self.TFRecord_path):  # TFRecord 파일이 존재하지 않은 경우
            print("<<< Making {} >>>".format(os.path.basename(self.TFRecord_path)))
            with tf.python_io.TFRecordWriter(self.TFRecord_path) as writer:  # TFRecord로 쓰자
                random.shuffle(self.file_path_list)
                for image_address in tqdm(self.file_path_list):
                    img_left, img_right = self.load_image(image_address)

                    '''넘파이 배열의 값을 바이트 스트링으로 변환한다.
                    tf.train.BytesList, tf.train.Int64List, tf.train.FloatList 을 지원한다.
                    '''
                    feature = \
                        {
                            'image_left': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_left.tostring())])),
                            'image_right': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_right.tostring())])),
                            'height': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[img_left.shape[0]])),
                            'width': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[img_left.shape[1]])),
                        }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    # 파일로 쓰자.
                    writer.write(example.SerializeToString())
            print("<<< Making {} is Completed >>>".format(os.path.basename(self.TFRecord_path)))
        else:  # TFRecord가 존재할 경우
            print("<<< {} already exists >>>".format(os.path.basename(self.TFRecord_path)))

    # tf.data.TFRecordDataset를 사용하는 방법 - TRRecord(이진 파일, 직렬화된 입력 데이터)라는 텐서플로우 표준 파일형식으로 저장된 파일을 불러와서 처리하기
    def Using_TFRecordDataset(self):

        # TFRecordDataset()사용해서 읽어오기
        dataset = tf.data.TFRecordDataset(self.TFRecord_path)
        dataset = dataset.map(self._image_preprocessing)
        dataset = dataset.shuffle(buffer_size=1000).repeat().batch(self.batch_size)
        # 사실 여기서 dataset.make_one_shot_iterator()을 사용해도 된다.
        iterator = dataset.make_initializable_iterator()
        # tf.python_io.tf_record_iterator는 무엇인가 ? TFRecord 파일에서 레코드를 읽을 수 있는 iterator이다.
        return iterator, iterator.get_next(), sum(1 for _ in tf.python_io.tf_record_iterator(self.TFRecord_path))


''' 
to reduce model oscillation [14], we follow
Shrivastava et al’s strategy [45] and update the discriminators
using a history of generated images rather than the ones
produced by the latest generative networks. We keep an image
buffer that stores the 50 previously generated images.

imagePool 클래스
# https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/utils.py 를 참고해서 변형했다.
'''


class ImagePool(object):

    def __init__(self, image_pool_size=50):

        self.image_pool_size = image_pool_size
        self.image_count = 0
        self.image_appender = []

    def __repr__(self):
        return "Image Pool class"

    def __call__(self, image=None):

        # 1. self.image_pool_size 사이즈가 0이거나 작으면, ImagePool을 사용하지 않는다.
        if self.image_pool_size <= 0:
            return image

        '''2. self.num_img 이 self.image_pool_size 보다 작으면, self.image_count을 하나씩 늘려주면서
        self.images_appender에 images를 추가해준다.
        self.image_pool_size 개 self.images_appender에 이전 images를 저장한다.'''
        if self.image_count < self.image_pool_size:
            self.image_appender.append(image)
            self.image_count += 1
            return image

        # copy에 대한 내용은 본 프로젝트의 copy_example.py를 참고!!!
        # np.random.rand()는 0~1 사이의 무작위 값을 출력한다.
        if np.random.rand() > 0.5:
            index = np.random.randint(low=0, high=self.image_pool_size, size=None)
            past_image = self.image_appender[index]
            self.image_appender[index] = image
            return past_image
        else:
            return image


if __name__ == "__main__":
    '''
    Dataset 은 아래에서 하나 고르자
    "cityscapes"
    "facades"
    "maps"
    '''
    dataset = Dataset(DB_name="facades", AtoB=True, batch_size=1, use_TrainDataset=True, inference_size=(256, 256))
    iterator, next_batch, data_length = dataset.iterator()

else:
    print("Dataset imported")
