lr = 0.003
Es_deep = []
for i in range(4):
    Es_deep.append(train_model(student, X_train, y_train, 50000, lr))
    #observe we reduce learning rate
    lr /= 3
Es_deep = np.array(Es_deep)
Es_deep = Es_deep.ravel()

# evaluate test error
loss_deep = compute_loss(student, X_test, y_test) / float(y_test.var())
print("Loss of deep student: ",loss_deep)
plot_loss(Es_deep)