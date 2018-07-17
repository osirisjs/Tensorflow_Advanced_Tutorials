from CNN import model

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
model(TEST=True, model_name="CNN", optimizer_selection="Adam", learning_rate=0.001, training_epochs=100,
      batch_size=256, display_step=1, batch_norm=True)
