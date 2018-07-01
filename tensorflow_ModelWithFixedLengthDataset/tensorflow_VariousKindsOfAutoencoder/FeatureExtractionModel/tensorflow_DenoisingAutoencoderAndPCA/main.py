import DenoisingAutoencoder as DA

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "Convolution_Autoencoder" or "Autoencoder"
DA.model(TEST=False, Comparison_with_PCA=True, model_name="Autoencoder",
                           corrupt_probability=0.5,
                           optimizer_selection="Adam", learning_rate=0.001, training_epochs=1, batch_size=256,
                           display_step=1, batch_norm=True)
