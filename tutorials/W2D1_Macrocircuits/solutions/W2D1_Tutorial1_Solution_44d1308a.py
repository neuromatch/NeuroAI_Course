torch.manual_seed(-1)

# Create teacher
n_in = 5     # input dimension
W_teacher, D_teacher = 5, 5  # teacher width, depth
sigma_teacher = 0.4     # teacher weight variance
teacher = make_MLP(n_in, W_teacher, D_teacher)
initialize_layers(teacher, sigma_teacher)

# generate train and test set
N_train, N_test = 4000, 1000
X_train, y_train = make_data(teacher, n_in, N_train)
X_test, y_test = make_data(teacher, n_in, N_test)