import ImageToImageTranslation as pix2pix

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
'''
distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
distamce_loss = None 일 경우 , 추가적인 loss 없음
'''
'''
내가 생각하는 Image-to-Image Translation with cGAN
1. generator 네트워크에 학습, 테스트시 둘다 dropout 적용  
2. discriminator에 patchGAN 적용 - texture/style loss로써 이해될수 있다고 함
'''

#네트워크의 구조는 논문에서 제시한 것과 동일합니다..
pix2pix.model(TEST=False, distance_loss="L2", optimizer_selection="Adam",
            beta1 = 0.9, beta2 = 0.999, # for Adam optimizer
            decay = 0.999, momentum = 0.9, # for RMSProp optimizer
            L2_weight=100,
            #batch_size는 1~10사이로 하자
            learning_rate=0.0002, training_epochs=10, batch_size=5, display_step=1)
