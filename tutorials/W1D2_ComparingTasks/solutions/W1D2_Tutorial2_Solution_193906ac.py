
# Dictionary to store normalized embeddings for each class
embeddings = {}
for i in range(10):
    embeddings[i] = test_embeddings_normed[test_labels == i]

# Within class cosine similarity:
for i in range(10):
    sims = embeddings[i] @ embeddings[i].T  # Compute cosine similarity matrix within the class
    np.fill_diagonal(sims, np.nan)  # Ignore diagonal values (self-similarity)
    cur_sim = np.nanmean(sims)  # Calculate the mean similarity excluding diagonal
    sim_matrix[i, i] = cur_sim  # Store the within-class similarity in the matrix

    print("Within class {} cosine similarity".format(i, cur_sim))

print("==================")

# Between class cosine similarity:
for i in range(10):
    for j in range(10):
        if i == j:
            continue  # Skip if same class (already computed)
        elif i > j:
            continue  # Skip if already computed (matrix symmetry)
        else:
            sims = embeddings[i] @ embeddings[j].T  # Compute cosine similarity between different classes
            cur_sim = np.mean(sims)  # Calculate the mean similarity
            sim_matrix[i, j] = cur_sim  # Store the similarity in the matrix
            sim_matrix[j, i] = cur_sim  # Ensure symmetry in the matrix
            print("{} and {} cosine similarity {}".format(i, j, cur_sim))

# Plotting the similarity matrix
plt.imshow(sim_matrix, vmin=0.0, vmax=1.0)
plt.title("Untrained Network Cosine Similarity Matrix")
plt.colorbar()
plt.show()