
sim_mat = np.zeros((4,4))

sim_mat[0,0] = spa.dot(circle, circle)
sim_mat[1,1] = spa.dot(square, square)
sim_mat[2,2] = spa.dot(triangle, triangle)
sim_mat[3,3] = spa.dot(shape, shape)

sim_mat[0,1] = sim_mat[1,0] = spa.dot(circle, square)
sim_mat[0,2] = sim_mat[2,0] = spa.dot(circle, triangle)
sim_mat[0,3] = sim_mat[3,0] = spa.dot(circle, shape)

sim_mat[1,2] = sim_mat[2,1] = spa.dot(square, triangle)
sim_mat[1,3] = sim_mat[3,1] = spa.dot(square, shape)
sim_mat[2,3] = sim_mat[3,2] = spa.dot(triangle, shape)