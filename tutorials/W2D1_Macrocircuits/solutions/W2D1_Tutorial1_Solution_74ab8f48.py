W_student, D_student = 10, 2  # student width, depth

lr = 1e-3
n_epochs = 20000
Es_shallow_train = np.zeros((len(Ws_student),n_epochs))
Es_shallow_test = np.zeros(len(Ws_student))

student = make_MLP(n_in, W_student, D_student)
initialize_layers(student, sigma_teacher)

# make sure we have enough data
P = get_num_params(n_in, W_student, D_student)
assert(N_train > 3*P)

# train
Es_shallow_train = train_model(student, X_train, y_train, n_epochs, lr, progressbar=True)

# # evaluate test error
Es_shallow_test = compute_loss(student, X_test, y_test)/float(y_test.var())
print('Shallow student loss: ',Es_shallow_test)
plot_students_predictions_vs_teacher_values(Es_shallow_train, X_test, y_test)