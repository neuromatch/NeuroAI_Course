D_student = 2  # student depth
Ws_student = np.array([5, 15, 45, 135]) # widths

lr = 1e-3
n_epochs = 20000
Es_shallow_train = np.zeros((len(Ws_student), n_epochs))
Es_shallow_test = np.zeros(len(Ws_student))


for index, W_student in enumerate(tqdm(Ws_student)):

    student = make_MLP(n_in, W_student, D_student)

    # make sure we have enough data
    P = get_num_params(n_in, W_student, D_student)
    assert(N_train > 3*P)

    # train
    Es_shallow_train[index] = train_model(student, X_train, y_train, n_epochs, lr, progressbar=False)
    Es_shallow_train[index] /= y_test.var()

    # evaluate test error
    loss = compute_loss(student, X_test, y_test)/y_test.var()
    Es_shallow_test[index] = loss

plot_loss_as_function_of_width(Ws_student, Es_shallow_test, Es_shallow_train)