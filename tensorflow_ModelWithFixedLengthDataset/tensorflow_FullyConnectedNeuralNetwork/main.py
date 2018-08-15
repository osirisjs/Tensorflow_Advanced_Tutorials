from FNN import model

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# Batch normalization은 Hidden Layer에만 추가합니다. 또한 활성화 함수전에 적용합니다.
model(TEST=True, model_name="FNN", optimizer_selection="Adam", learning_rate=0.001, training_epochs=50,
      batch_size=256, display_step=1, batch_norm=True)
