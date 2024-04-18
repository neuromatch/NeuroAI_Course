error_target = 1e-6

m,b = np.polyfit(np.log(Ws_student), np.log(Es_shallow_test), 1)
print('Predicted width: ',np.exp((np.log(error_target) - b) / m))