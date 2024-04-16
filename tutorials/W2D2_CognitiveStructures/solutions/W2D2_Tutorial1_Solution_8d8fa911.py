
sim_mat = np.zeros((4,4))

sim_mat[0,0] = (circle | circle).item()
sim_mat[1,1] = (square | square).item()
sim_mat[2,2] = (triangle | triangle).item()
sim_mat[3,3] = (shape | shape).item()

sim_mat[0,1] = sim_mat[1,0] = (circle | square).item()
sim_mat[0,2] = sim_mat[2,0] = (circle | triangle).item()
sim_mat[0,3] = sim_mat[3,0] = (circle | shape).item()

sim_mat[1,2] = sim_mat[2,1] = (square | triangle).item()
sim_mat[1,3] = sim_mat[3,1] = (square | shape).item()

sim_mat[2,3] = sim_mat[3,2] = (triangle | shape).item()