
class WeightPerturbMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through weight perturbation
    """

    def perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for perturbation learning, using noise with SD as given
        """

        # get the random perturbations
        delta_W_h = rng.normal(scale=noise, size=self.W_h.shape)
        delta_W_y = rng.normal(scale=noise, size=self.W_y.shape)

        # calculate the loss with and without the perturbations
        loss_now = self.mse_loss(rng, inputs, targets)
        loss_per = self.mse_loss(rng, inputs, targets, self.W_h + delta_W_h, self.W_y + delta_W_y)

        # updates
        delta_loss = loss_now - loss_per
        W_h_update = delta_loss * delta_W_h / noise ** 2
        W_y_update = delta_loss * delta_W_y / noise ** 2
        return W_h_update, W_y_update