from model import model

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
model(TEST=True, optimizer_selection="Adam", learning_rate=0.0009, training_epochs=300000, batch_size=256,
      display_step=100,
      # 전 회차 당첨번호 6자리 입력
      # 반드시 이차원 배열로 선언
      previous_first_prize_number=[[11, 30, 34, 35, 42, 44]], number_of_prediction=5)
