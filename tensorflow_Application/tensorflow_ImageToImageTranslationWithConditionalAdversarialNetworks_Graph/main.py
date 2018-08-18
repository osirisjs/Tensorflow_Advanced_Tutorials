from tensorflow.python.client import device_lib

import ImageToImageTranslation as pix2pix

'''
1. 설명
데이터셋 다운로드는 - https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/ 에서 <데이터셋> 을 내려 받자. 
논문에서 추천하는 hyperparameter 는 200 epoch, 1 ~ 10 batch size 정도, beta1 = 0.5, beta2=0.999, lr=0.0002

입력 크기 : <256x256x3>
출력 크기 : <256x256x3>
optimizers_ selection = "Adam" or "RMSP" or "SGD"
batch_size = 1 -> instance norm, batch_size > 1 -> batch_norm
저자코드 - https://github.com/phillipi/pix2pix/blob/master/models.lua
generator는 unet을 사용한다.
discriminator의 구조는 PatchGAN 70X70을 사용한다. 

Training, Test Generator의 동작 방식이 같다. - 드롭아웃 적용, batchnorm 이동 평균 안씀 - 그래도 옵션으로 주자
At inference time, we run the generator net in exactly
the same manner as during the training phase. This differs
from the usual protocol in that we apply dropout at test time,
and we apply batch normalization [28] using the statistics of
the test batch, rather than aggregated statistics of the training
batch. This approach to batch normalization, when the
batch size is set to 1, has been termed “instance normalization”
and has been demonstrated to be effective at image
generation tasks [53]. In our experiments, we use batch
sizes between 1 and 10 depending on the experiment

-논문 내용과 똑같이 구현했다.

2. loss에 대한 옵션
distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
distamce_loss = None 일 경우 , 추가적인 loss 없음

3. pathGAN?
patchGAN 은 논문의 저자가 붙인이름이다. - ReceptiveField 에 대한 이해가 필요하다. -> 이 내용에 대해 혼돈이 있을 수 있으니 ref폴더의 안의 receptiveField 내용과 receptiveFieldArithmetic폴더의 receptiveField 크기 구현 코드를 참고한다.
저자의 답변이다.(깃허브의 Issues에서 찾았다.)
This is because the (default) discriminator is a "PatchGAN" (Section 2.2.2 in the paper).
This discriminator slides across the generated image, convolutionally, 
trying to classify if each overlapping 70x70 patch is real or fake. 
This results in a 30x30 grid of classifier outputs, 
each corresponding to a different patch in the generated image.
'''

'''
DB_name 은 아래에서 하나 고르자
1. "cityscapes" - 요것만 segmentation -> image 생성이다.
2. "facades"
3. "maps"

AtoB -> A : image,  B : segmentation
AtoB = True  -> image -> segmentation
AtoB = False -> segmentation -> image
'''

'''
텐서플로우의 GPU 메모리정책 - GPU가 여러개 있으면 기본으로 메모리 다 잡는다. -> 모든 GPU 메모리를 다 잡는다. - 원하는 GPU만 쓰고 싶은 해결책은 CUDA_VISIBLE_DEVICES 에 있다. - 83번째 라인을 읽어라!!!
Allowing GPU memory growth
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process.
This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory, 
or to only grow the memory usage as is needed by the process. TensorFlow provides two Config options on the Session to control this.

The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations: 
it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. 
Note that we do not release memory, since that can lead to even worse memory fragmentation. To turn this option on, set the option in the ConfigProto by:

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
The second method is the per_process_gpu_memory_fraction option, which determines the fraction of the overall amount of memory that each visible GPU should be allocated. 
For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
'''

print("<<< * 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항 >>>")
print(
    "<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"번호\"와 os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"를 tf API를 하나라도 사용하기 전에 작성해 넣으면 됨 >>>")
print(
    "<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES = 0(gpu 번호) python main,py 을 터미널 창에 적고 ENTER - Ubuntu에서만 동작 >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1'과 같은)를 적어 넣으면 됨 >>>\n")

# 특정 GPU로 학습 하고 싶을때, 아래의 2줄을 꼭 써주자.(Ubuntu , window 둘 다 가능) - 반드시 Tensorflow의 API를 하나라도 쓰기 전에 아래의 2줄을 입력하라!!!
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 현재 사용하고 있는 GPU 번호를 얻기 위한 코드 - 여러개의 GPU를 쓸 경우 정보 확인을 위해!
print("<<< Ubuntu Terminal 창에서 지정해준 경우, 무조건 GPU : 1대, GPU 번호 : 0 라고 출력 됨 >>>")
local_device_protos = device_lib.list_local_devices()
GPU_List = [x.name for x in local_device_protos if x.device_type == 'GPU']
# gpu_number_list = []
print("<<< # 사용 가능한 GPU : {} 대 >>>".format(len(GPU_List)))
print("<<< # 사용 가능한 GPU 번호 : >>>", end="")
for i, GL in enumerate(GPU_List):
    num = GL.split(":")[-1]
    # gpu_number_list.append(num)
    if len(GPU_List) - 1 == i:
        print(" " + num)
    else:
        print(" " + num + ",", end="")

''' 
256x256 크기 이상의 다양한 크기의 이미지를 동시 학습 하는 것이 가능하다. - 이것을 구현하는데 생각보다 시간이 오래 걸렸다.
데이터가 쌍으로 존재하다보니 반반으로 나눠야 한다. 심지어 다양한 크기의 데이터가 쌍이다. 그래프는 입 출력 등의 모양만 알고있는 채로 그려진다.
자기가 알아서 나눌수가 없다. 무조건 정보를 줘야한다. 즉 그래프는 나누는 포인트정보가 필요하다.
이게 사실 pytorch, gluon과 같은 imperative 언어였다면, 실행하면서 계산이 가능하므로 생각할 필요도 없는 문젠데, 
symbolic 언어인 텐서플로에서는 연산그래프가 고정되어버리기 때문에 복잡하다.
(numpy로 Dataset을 구현 하고, placeholder에 feed dict 으로 넣어줬으면, 금방 끝났을 일이었지만, 텐서플로에 대한 이해가 조금 더 깊어졌다.)

내가 생각한 총 3가지 방법이 있었다. 
첫번째, tf.data.Dataset을 batch, shuffle등의 전처리 기능으로만 쓰고, sess.run()으로 실행한 후 numpy로 그림을 나눠줘서 학습하는 방법 
 - imperative 방식과 다를게 없다. 또한 텐서플로를 제대로 사용하지 않는 것임!!! 
 - 이럴꺼면 numpy로 

두번째 DB에 대해 나누는 포인트 정보를 가지고 있는 List파일을 만들어서, tf.data.Dataset.from_tensor_slices에서 같이 불러오는 방법 
 - 비교적 텐서플로답게 사용하는 것이긴 하지만 최선은 아니다. 
 
세번째 방법 TFRecord 이용하는 방식 -> TFRecord로 DB를 쓸 때 내가 원하는 정보를 포함해서 쓸 수 있고, 내가 원하는 정보를 불러오는게 가능하다. 이게 바로 텐서플로다.'''

# TEST=False 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
# TEST=True 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.
# optimizers_ selection = "Adam" or "RMSP" or "SGD"
pix2pix.model(TEST=False, TFRecord=True, filter_size=16, AtoB=False, DB_name="facades",
              norm_selection ="BN", #IN - instance normalizaiton , BN -> batch normalization, NOTHING
              distance_loss="L1",
              distance_loss_weight=100, optimizer_selection="Adam",
              beta1=0.5, beta2=0.999,  # for Adam optimizer
              decay=0.999, momentum=0.9,  # for RMSProp optimizer
              # batch_size는 1~10사이로 하자
              image_pool=True,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
              image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지?
              learning_rate=0.0002, training_epochs=1, batch_size=1, display_step=1, Dropout_rate=0.5,
              inference_size=(256, 256),  # TEST=True 일 떄, inference할 크기는 256 x 256 이상이어야 한다.
              # using_moving_variable - 이동 평균, 이동 분산을 사용할지 말지 결정하는 변수 - 논문에서는 Test = Training
              # 후에 moving_variable을 사용할 수도 있을 경우를 대비하여 만들어 놓은 변수 Test=False일 때
              using_moving_variable=False,  # TEST=True 일때, Moving Average를 사용할건지 말건지 선택하는 변수 -> 보통 사용안함.
              # 아래의 변수가 True이면 그래프만 그리고 종료,
              only_draw_graph=False, # TEST=False 일 떄, 그래프만 그리고 종료할지 말지
              show_translated_image=True,  # TEST=True 일 때변환 된 이미지를 보여줄지 말지
              weights_to_numpy=True,  # TEST=True 일 때 가중치를 npy 파일로 저장할지 말지
              save_path="translated_image")  # TEST=True 일 때 변환된 이미지가 저장될 폴더
