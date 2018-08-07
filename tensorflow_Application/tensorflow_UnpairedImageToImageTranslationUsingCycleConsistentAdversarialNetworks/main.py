from tensorflow.python.client import device_lib
import os
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

# 현재 사용하고 있는 GPU 번호를 얻기 위한 코드 - 여러개의 GPU를 쓸 경우 정보확인을 위해!
print("Terminal or CMD 창에서 지정해준 경우, 무조건 GPU : 1대, GPU 번호 : 0 라고 출력됨") 
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

print("* 한대의 컴퓨터에 여러대의 GPU 가 설치되어 있을 경우 참고할 사항")
print("<<< 경우의 수 1 : GPU가 여러대 설치 / 통합개발 환경에서 실행 / GPU 번호 지정 원하는 경우 -> os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"번호\"와 os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"를 pix2pix.model() 호출 전에 작성해 넣으면 됨 >>>")
print(
    "<<< 경우의 수 2 : GPU가 여러대 설치 / 터미널 창에서 실행 / GPU 번호 지정 원하는 경우  -> CUDA_VISIBLE_DEVICES = 0(gpu 번호) python main,py 을 터미널 창에 적고 ENTER >>>")
print("<<< CPU만 사용하고 싶다면? '현재 사용 가능한 GPU 번호' 에 없는 번호('-1'과 같은)를 적어 넣으면 됨 >>>\n")

#특정 GPU로 학습 하고 싶을때, 아래의 2줄을 꼭 써주자.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DB_name = "horse2zebra" 만...
cycleGAN.model(TEST=False, DB_name="horse2zebra", use_TFRecord=True, cycle_consistency_loss="L1",
               cycle_consistency_loss_weight=10,
               optimizer_selection="Adam", beta1=0.9, beta2=0.999,  # for Adam optimizer
               decay=0.999, momentum=0.9,  # for RMSProp optimizer
               use_identity_mapping=True,  # 논문에서는 painting -> photo DB 로 네트워크를 학습할 때 사용했다고 함.
               norm_selection="instance_norm",  # "instance_norm" or nothing
               image_pool=True,  # discriminator 업데이트시 이전에 generator로 부터 생성된 이미지의 사용 여부
               image_pool_size=50,  # image_pool=True 라면 몇개를 사용 할지? 논문에선 50개 사용했다고 나옴.
               learning_rate=0.0002, training_epochs=200, batch_size=1, display_step=1,
               # 학습 완료 후 변환된 이미지가 저장될 폴더 2개가 생성 된다. AtoB_translated_image , BtoA_translated_image 가 붙는다.
               save_path="translated_image")
