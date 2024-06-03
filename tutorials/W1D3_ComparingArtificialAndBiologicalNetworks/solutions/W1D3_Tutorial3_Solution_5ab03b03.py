
# Compute rdms for each layer of AlexNet
alexnet_rdms_dict = {}
for layer, dataset in alexnet_datasets.items():
    alexnet_rdms_dict[layer] = rsa.rdm.calc_rdm(dataset)