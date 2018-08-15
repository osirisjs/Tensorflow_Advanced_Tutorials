import SparseAutoencoder as SA

# optimizers_ selection = "Adam" or "RMSP" or "SGD"
# model_name = "Convolution_Autoencoder" or "Autoencoder"
SA.model(TEST=True, Comparison_with_PCA=True, model_name="Autoencoder", target_sparsity=0.2, weight_sparsity=0.1,
         optimizer_selection="Adam", learning_rate=0.001, training_epochs=1, batch_size=512, display_step=1, batch_norm=True)
