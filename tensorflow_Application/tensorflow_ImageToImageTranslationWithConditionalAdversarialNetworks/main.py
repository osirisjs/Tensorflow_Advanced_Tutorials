import ImageToImageTranslation as pix2pix

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "GAN"

pix2pix.model(TEST=True, noise_size=128,optimizer_selection="Adam", learning_rate=0.0002, training_epochs=100,
                                    batch_size=128,
                                    display_step=1, batch_norm=False)
