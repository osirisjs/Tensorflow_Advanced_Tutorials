import ImageToImageTranslation as pix2pix
from tensorflow.python.client import device_lib
'''
1. 설명
데이터셋 다운로드는 - https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/ 에서 <데이터셋> 을 내려 받자. 
논문에서 추천하는 hyperparameter 는 200 epoch, 1 ~ 10 batch size 정도, beta1 = 0.5, beta2=0.999, lr=0.0002

입력 크기 : <256x256x3>
출력 크기 : <256x256x3>
optimizers_ selection = "Adam" or "RMSP" or "SGD"
batch_size = 1 -> instance norm, batch_size > 1 -> batch_norm
저자코드 따라함 - https://github.com/phillipi/pix2pix/blob/master/models.lua
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

use_TFRecord 은 데이터셋을 텐서플로우 표준 파일형식(TFRecord)으로 변환해서 사용할건지?

AtoB -> A : image,  B : segmentation
AtoB = True  -> image -> segmentation
AtoB = False -> segmentation -> image
'''

# 현재 사용하고 있는 GPU 번호를 얻기 위한 코드 - 여러개의 GPU를 쓸 경우 정보확인을 위해!
print("Terminal or CMD 창에서 지정해준 경우, 무조건 GPU : 1대, GPU 번호 : 0 라고 출력됨") 
local_device_protos = device_lib.list_local_devices()
GPU_List = [x.name for x in local_device_protos if x.device_type == 'GPU']
# gpu_number_list = []
print("# 사용 가능한 GPU : {} 대".format(len(GPU_List)))
print("# 사용 가능한 GPU 번호 :", end="")
for i, GL in enumerate(GPU_List):
    num = GL.split(":")[-1]
    # gpu_number_list.append(num)
    if len(GPU_List)-1 == i:
        print(" "+num)
    else:
        print(" " + num + ",", end="")

print("* 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항")
print("<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"번호\"와 os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"를 pix2pix.model() 호출 전에 작성해 넣으면 됨 >>>")
print(
    "<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES = 0(gpu 번호) python main,py 을 터미널 창에 적고 ENTER >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1'과 같은)를 적어 넣으면 됨 >>>\n")

#특정 GPU로 학습 하고 싶을때, 아래의 2줄을 꼭 써주자.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pix2pix.model(TEST=True, AtoB= True, DB_name="facades", use_TFRecord=True, distance_loss="L1",
              distance_loss_weight=100, optimizer_selection="Adam",
              beta1=0.5, beta2=0.999,  # for Adam optimizer
              decay=0.999, momentum=0.9,  # for RMSProp optimizer
              # batch_size는 1~10사이로 하자
              image_pool=True,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
              image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지?
              learning_rate=0.0002, training_epochs=200, batch_size=1, display_step=1, Dropout_rate=0.5,
              using_moving_variable=False,  # using_moving_variable - 이동 평균, 이동 분산을 사용할지 말지 결정하는 변수
              save_path="translated_image")  # 학습 완료 후 변환된 이미지가 저장될 폴더
