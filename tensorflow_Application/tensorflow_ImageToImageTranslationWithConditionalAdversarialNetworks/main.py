import ImageToImageTranslation as pix2pix

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
'''
distance_loss = 'L1' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L1 loss를 생성
distance_loss = 'L2' 일 경우 , generator에서 나오는 출력과 실제 출력값을 비교하는 L2 loss를 생성
distamce_loss = None 일 경우 , 추가적인 loss 없음
'''
pix2pix.model(TEST=False, distance_loss="L2", optimizer_selection="Adam",
              learning_rate=0.0002, training_epochs=10, batch_size=128, display_step=1)
