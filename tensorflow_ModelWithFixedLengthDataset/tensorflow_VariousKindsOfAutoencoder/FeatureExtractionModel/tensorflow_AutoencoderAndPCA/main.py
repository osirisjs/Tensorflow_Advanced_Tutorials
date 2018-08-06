import Autoencoder

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "Convolution_Autoencoder" or "Autoencoder"
Autoencoder.model(TEST=False, Comparison_with_PCA=True, model_name="Autoencoder",
                  optimizer_selection="Adam", learning_rate=0.001, training_epochs=1, batch_size=512,
                  display_step=1, batch_norm=True)
