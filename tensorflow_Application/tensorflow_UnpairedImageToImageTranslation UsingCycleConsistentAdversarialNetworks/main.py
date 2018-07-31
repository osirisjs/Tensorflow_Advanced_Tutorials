import UnpairedImageToImageTranslation as cycleGAN
from tensorflow.python.client import device_lib
'''
간단한 설명
데이터셋 다운로드는 - https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/ 에서 <데이터셋> 을 내려 받자. 
논문에서 추천하는 hyperparameter 는 200 epoch, 1 ~ 10 batch size 정도, beta1 = 0.5, beta2=0.999, lr=0.0002
입력 크기 : <256x256x3>
출력 크기 : <256x256x3>
optimizers_ selection = "Adam" or "RMSP" or "SGD"
batch_size = 1 -> instance norm, batch_size > 1 -> batch_norm
저자코드 따라함 - https://github.com/phillipi/pix2pix/blob/master/models.lua
G_generator, F_generator 는 residual net 을 사용한다.()
discriminator의 구조는 PatchGAN 70X70을 사용한다. 
-논문 내용과 똑같이 구현했다.
'''

# 현재 사용하고 있는 GPU 번호를 얻기 위한 코드입니다. - 제가 여러개의 GPU를 써서..
local_device_protos = device_lib.list_local_devices()
GPU_List = [x.name for x in local_device_protos if x.device_type == 'GPU']
# gpu_number_list = []
print("# 사용 가능한 GPU : {}대".format(len(GPU_List)))
print("# 사용 가능한 GPU 번호 :", end="")
for i, GL in enumerate(GPU_List):
    num = GL.split(":")[-1]
    # gpu_number_list.append(num)
    if len(GPU_List)-1 == i:
        print(" "+num)
    else:
        print(" "+num+",")

print("* 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항")
print("<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> "'os.environ["CUDA_VISIBLE_DEVICES"]'"를 pix2pix.model() 호출 전에 작성해 넣으면 됨 >>>")
print("<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES = 0(gpu 번호) python main,py 을 터미널 창에 적고 ENTER >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1' 이라던지)를 적어 넣으면 됨 >>>\n")

cycleGAN.model(TEST=False, AtoB= True, DB_name="maps", use_TFRecord=True, distance_loss="L1",
              distance_loss_weight=100, optimizer_selection="Adam",
              beta1=0.5, beta2=0.999,  # for Adam optimizer
              decay=0.999, momentum=0.9,  # for RMSProp optimizer
              # batch_size는 1~10사이로 하자
              learning_rate=0.0002, training_epochs=200, batch_size=1, display_step=1, Dropout_rate=0.5,
              using_moving_variable=False,  # using_moving_variable - 이동 평균, 이동 분산을 사용할지 말지 결정하는 변수
              save_path="translated_image")  # 학습 완료 후 변환된 이미지가 저장될 폴더
