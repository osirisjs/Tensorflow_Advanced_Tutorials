import DenoisingAutoencoder as DA

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name =  CDA -> ConvolutionDenosingAutoencoder" or DA -> DenosingAutoencoder
DA.model(TEST=True, Comparison_with_PCA=True, model_name="DA",
         corrupt_probability=0.5,
         optimizer_selection="Adam", learning_rate=0.001, training_epochs=3, batch_size=256,
         display_step=1, batch_norm=True)
