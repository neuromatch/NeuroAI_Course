
shape_sim_mat = np.zeros((4,4))

shape_sim_mat[0,0] = (circle | circle).item()
shape_sim_mat[1,1] = (square | square).item()
shape_sim_mat[2,2] = (triangle | triangle).item()
shape_sim_mat[3,3] = (shape | shape).item()

shape_sim_mat[0,1] = shape_sim_mat[1,0] = (circle | square).item()
shape_sim_mat[0,2] = shape_sim_mat[2,0] = (circle | triangle).item()
shape_sim_mat[0,3] = shape_sim_mat[3,0] = (circle | shape).item()

shape_sim_mat[1,2] = shape_sim_mat[2,1] = (square | triangle).item()
shape_sim_mat[1,3] = shape_sim_mat[3,1] = (square | shape).item()

shape_sim_mat[2,3] = shape_sim_mat[3,2] = (triangle | shape).item()