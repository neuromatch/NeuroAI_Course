
# Dictionary to store normalized embeddings for each class
embeddings = {}
for i in range(10):
    embeddings[i] = test_embeddings_untrained[test_labels_untrained == i]

# Within class cosine similarity:
for i in range(10):
    sims = embeddings[i] @ embeddings[i].T  # Compute cosine similarity matrix within the class
    np.fill_diagonal(sims, np.nan)  # Ignore diagonal values (self-similarity)
    cur_sim = np.nanmean(sims)  # Calculate the mean similarity excluding diagonal
    sim_matrix[i, i] = cur_sim  # Store the within-class similarity in the matrix

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

plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, vmin=0.0, vmax=1.0, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
plt.title("Untrained Network Cosine Similarity Matrix")
plt.show()