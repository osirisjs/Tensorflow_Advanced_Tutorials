from tensorflow.python.client import device_lib

import UnpairedImageToImageTranslation as cycleGAN

'''
간단한 설명
데이터셋 다운로드는 - https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip 에서 <데이터셋> 을 내려 받자. 
논문에서 추천하는 batch size = 1, Adam Optimizer, lr=0.0002
입력 크기 : <256x256x3>
출력 크기 : <256x256x3>
optimizers_ selection = "Adam" or "RMSP" or "SGD"
AtoB_generator, BtoA_generator 는 residual net 을 사용한다. -  9 blocks 
discriminator의 구조는 PatchGAN 70X70을 사용한다. 
-논문 내용과 거의 똑같이 구현했다. - image pool도 구현하기!!!
'''

'''
텐서플로우의 GPU 메모리정책 - GPU가 여러개 있으면 기본으로 메모리 다 잡는다. -> 아래의 옵션을 써도, 모든 GPU 메모리를 다 잡는다. - 원하는 GPU만 쓰고 싶은 해결책은 CUDA_VISIBLE_DEVICES 에 있다. - 41번째 라인을 읽어라!!!
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

print("* 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항")
print(
    "<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"번호\"와 os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"를 tf API를 하나라도 사용하기 전에 작성해 넣으면 됨 >>>")
print(
    "<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES = 0(gpu 번호) python main,py 을 터미널 창에 적고 ENTER - Ubuntu에서만 동작 >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1'과 같은)를 적어 넣으면 됨 >>>\n")

# 특정 GPU로 학습 하고 싶을때, 아래의 2줄을 꼭 써주자.(Ubuntu , window 둘 다 가능) - 반드시 Tensorflow의 API를 하나라도 쓰기 전에 아래의 2줄을 입력하라!!!
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 현재 사용하고 있는 GPU 번호를 얻기 위한 코드 - 여러개의 GPU를 쓸 경우 정보 확인을 위해!
print("Ubuntu Terminal 창에서 지정해준 경우, 무조건 GPU : 1대, GPU 번호 : 0 라고 출력 됨")
local_device_protos = device_lib.list_local_devices()
GPU_List = [x.name for x in local_device_protos if x.device_type == 'GPU']
# gpu_number_list = []
print("# 사용 가능한 GPU : {}대".format(len(GPU_List)))
print("# 사용 가능한 GPU 번호 :", end="")
for i, GL in enumerate(GPU_List):
    num = GL.split(":")[-1]
    # gpu_number_list.append(num)
    if len(GPU_List) - 1 == i:
        print(" " + num)
    else:
        print(" " + num + ",", end="")

# DB_name = "horse2zebra" 만...
# 256x256 크기 이상의 다양한 크기의 이미지를 동시 학습 하는 것이 가능하다
# TEST=False 시 입력 이미지의 크기가 256x256 미만이면 강제 종료한다.- 관련 코드는 UnpairedImageToImageTranslation.py 의 458번줄을 보라.
# optimizers_ selection = "Adam" or "RMSP" or "SGD"
cycleGAN.model(TEST=True, DB_name="horse2zebra", use_TFRecord=True, cycle_consistency_loss="L1",
               cycle_consistency_loss_weight=10,
               optimizer_selection="Adam", beta1=0.5, beta2=0.999,  # for Adam optimizer
               decay=0.999, momentum=0.9,  # for RMSProp optimizer
               use_identity_mapping=False,  # 논문에서는 painting -> photo DB 로 네트워크를 학습할 때 사용 - 우선은 False
               norm_selection="instancenorm",  # "instancenorm" or nothing
               image_pool=False,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
               image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지? 논문에선 50개 사용했다고 나옴.
               learning_rate=0.0002, training_epochs=1, batch_size=1, display_step=1,
               weight_decay_epoch=100,  # 몇 epoch 뒤에 learning_rate를 줄일지
               learning_rate_decay=0.99,  # learning_rate를 얼마나 줄일지
               only_draw_graph=False,  # TEST=False 일 떄, 그래프만 그리고 종료할지 말지
               inference_size = (512, 512), # TEST=True 일 떄, inference할 크기는 256 x 256 이상이어야 한다. - 관련 코드는 Dataset.py 의 64번째 줄
               # 학습 완료 후 변환된 이미지가 저장될 폴더 2개가 생성 된다.(폴더 2개 이름 -> AtoB_translated_image , BtoA_translated_image )
               save_path="translated_image")  # TEST=True 일 때 변환된 이미지가 저장될 폴더
