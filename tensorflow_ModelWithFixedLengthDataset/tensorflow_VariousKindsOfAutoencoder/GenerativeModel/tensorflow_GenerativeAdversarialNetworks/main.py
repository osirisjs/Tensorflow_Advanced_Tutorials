import GenerativeAdversarialNetworks

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "GAN"
# 원하는 숫자를 생성이 가능하다.
# 흑백 사진(Generator의 입력으로) -> 컬러(Discriminator의 입력)로 만들기
# 선화(Generator의 입력)를 (여기서 채색된 선화는 Discriminator의 입력이 된다.)채색가능
'''
targeting = False 일 때는 숫자를 무작위로 생성하는 GAN MODEL 생성 - General GAN
targeting = True 일 때는 숫자를 타게팅 하여 생성하는 GAN MODEL 생성 - Conditional GAN

model_name은 붙이고 싶은 것 붙이면 된다.  
'''
GenerativeAdversarialNetworks.model(TEST=True, noise_size=128, targeting=True,
                                    optimizer_selection="Adam", learning_rate=0.0002, training_epochs=100,
                                    batch_size=128,
                                    #batch_norm을 쓰면 생성이 잘 안된다는..
                                    display_step=1, batch_norm=False)
