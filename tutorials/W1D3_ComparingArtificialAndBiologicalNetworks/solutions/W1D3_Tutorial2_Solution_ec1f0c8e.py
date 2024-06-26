
return_layers = ['input', 'conv1', 'conv2', 'fc1', 'fc2']
imgs, labels = sample_images(test_loader, n=50) # grab 500 samples from the test set
model_features = extract_features(model_robust, imgs.to(device), return_layers)

plot_dim_reduction(model_features, labels, transformer_funcs =['MDS'])