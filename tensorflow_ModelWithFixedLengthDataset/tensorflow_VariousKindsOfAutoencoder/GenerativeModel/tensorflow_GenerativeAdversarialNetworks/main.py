import GenerativeAdversarialNetworks

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# 원하는 숫자를 생성이 가능하다.
# 흑백 사진(Generator의 입력으로) -> 컬러(Discriminator의 입력)로 만들기
# 선화(Generator의 입력)를 (여기서 채색된 선화는 Discriminator의 입력이 된다.)채색가능
'''
targeting = False 일 때는 숫자를 무작위로 생성하는 GAN MODEL 생성 - General GAN
targeting = True 일 때는 숫자를 타게팅 하여 생성하는 GAN MODEL 생성 - Conditional GAN 

targeting = True 일 때 -> distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
targeting = True 일 때 -> distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
targeting = True 일 때 -> distamce_loss = None 일 경우 , 추가적인 loss 없음
참고 : distance_loss를 사용하지 않고, batch_norm을 쓰면 생성이 잘 안된다. 네트워크 구조를 간단히 하기위해
fully connected network를 사용해서 그런지 batch_norm이 generator가 숫자이미지를 생성하려는 것을 방해하는 것 같다.
'''
GenerativeAdversarialNetworks.model(TEST=True, noise_size=128, targeting=True, distance_loss="L1",
                                    distance_loss_weight=1, \
                                    optimizer_selection="Adam", learning_rate=0.0002, training_epochs=20,
                                    batch_size=128,
                                    display_step=1, batch_norm=True)
