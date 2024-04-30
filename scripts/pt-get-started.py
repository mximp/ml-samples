import torch
import matplotlib.pyplot as plt

data = torch.arange(-5, 5, 0.1).view(-1, 1)

func = -5 * data

y = func + 0.4 * torch.randn(data.size())


# defining the function for forward pass for prediction
def model_func(x):
    return w * x


# evaluating data points with Mean Square Error
def err_func(y_pred, y_act):
    return torch.mean((y_pred - y_act) ** 2)


w = torch.tensor(-10.0, requires_grad=True)

step_size = 0.1
loss_list = []
iterations = 20

for i in range(iterations):
    # making predictions with forward pass
    Y_pred = model_func(data)
    # calculating the loss between original and predicted data points
    loss = err_func(Y_pred, y)
    # storing the calculated loss in a list
    loss_list.append(loss.item())
    # backward pass for computing the gradients of the loss w.r.t to
    # learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    w.data = w.data - step_size * w.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    # printing the values for understanding
    print('{},\t{},\t{}'.format(i, loss.item(), w.item()))

# Plotting the loss after each iteration
plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid(True, color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.show()
