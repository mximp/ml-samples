import torch
import torch.nn as nn
import torch.optim as optim

# Sample data: y = 3x + 2
x_train = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y_train = torch.tensor([[5.0], [8.0], [11.0]], requires_grad=False)


# Step 2: Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        return self.linear(x)


# Create a model instance
model = LinearRegressionModel()

# Step 3: Define loss function
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Step 4: Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Step 5: Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute the loss
    loss = criterion(y_pred, y_train)

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# After training
print('Training complete')

# Example of using the model for prediction
with torch.no_grad():  # We don't need gradients for prediction
    new_x = torch.tensor([[4.0]])
    predicted = model(new_x)
    print(f'Prediction for input {new_x.item()}: {predicted.item()}')
