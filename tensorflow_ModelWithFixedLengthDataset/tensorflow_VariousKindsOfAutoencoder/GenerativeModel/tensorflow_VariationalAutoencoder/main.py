import VariationalAutoencoder as VA

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "Convolution_Autoencoder" or "Autoencoder"
# latent_number는 2의 배수인 양수여야 한다.
VA.model(TEST=False, latent_number=32, model_name="Autoencoder", optimizer_selection="Adam", \
         learning_rate=0.001, training_epochs=1, batch_size=512, display_step=1, batch_norm=True)
