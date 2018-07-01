from model import model

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
model(TEST=True, optimizer_selection="Adam", learning_rate=0.0009, training_epochs=60000, batch_size=256, display_step=100,
      # 전 회차 당첨번호 6자리 입력
      previous_first_prize_number=[1, 3, 12, 14, 16, 43], number_of_prediction=10)
