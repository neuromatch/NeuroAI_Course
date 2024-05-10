
def calculate_mean_absolute_strength(model):
    # Calculate the mean absolute connection strength of the recurrent weight matrix
    return torch.mean(torch.abs(model.J)).item()

def perturb_recurrent_weights(model, mean_strength, perturbation_percentage):
    perturbation_strength = mean_strength * perturbation_percentage
    with torch.no_grad():
        noise = torch.randn_like(model.J) * perturbation_strength
        perturbed_weights = model.J + noise
        return perturbed_weights

def test_perturbed_structure(model, perturbation_percentages, test_loader, criterion, device, max_error):
    model.eval()  # Set the model to evaluation mode
    mean_strength = calculate_mean_absolute_strength(model)
    perturbation_results = []  # List to store (mean error, std dev) tuples

    original_weights = model.J.data.clone()  # Save the original weights

    for percentage in perturbation_percentages:
        multiple_perturbations_error = []
        print(f"Testing perturbation percentage {percentage:.4f}")

        for perturbation in tqdm(range(30)):  # Perturb 30 times for each strength
            batch_errors = []
            perturbed_weights = perturb_recurrent_weights(model, mean_strength, percentage)
            model.J.data = perturbed_weights.data

            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                h = model.init_hidden(batch_size).to(device)

                outputs = torch.zeros_like(targets).to(device)
                for t in range(inputs.shape[1]):
                    output, h, *rest = model(inputs[:, t, :], h)
                    outputs[:, t, :] = output

                loss = criterion(outputs, targets).item()
                batch_errors.append(loss)

            # Reset to original weights after each perturbation
            model.J.data = original_weights.data
            multiple_perturbations_error.append(np.mean(batch_errors))

        mean_error = np.mean(multiple_perturbations_error)  # Average over the 50 perturbations
        std_dev_error = np.std(multiple_perturbations_error)  # Standard deviation for error bars
        perturbation_results.append((100 * mean_error / max_error, 100 * std_dev_error / max_error))

        # Normalize the errors
        print(f"Completed testing for perturbation percentage {percentage:.4f}. Mean error: {mean_error:.4f}, Std. dev.: {std_dev_error:.4f}\n")

    return perturbation_results