import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate(model, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_data)
    return output

# Define the first MLP model
class MLP_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the second MLP model
class MLP_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the third MLP model
class MLP_3(nn.Module):
    def __init__(self, mlp_1, mlp_2):
        super(MLP_3, self).__init__()
        self.mlp_1 = mlp_1
        self.mlp_2 = mlp_2

    def forward(self, x):
        # Forward pass for model_2 to get the parameter modifier
        param_modifier = self.mlp_2(x)

        # In-place modification of model_1 parameters using parameter modifier
        modified_params = [param + param_modifier.mean() for param in self.mlp_1.parameters()]

        # Apply modified parameters to model_1
        idx = 0
        for param in self.mlp_1.parameters():
            param.data = modified_params[idx].data
            idx += 1

        # Forward pass for model_1 with modified parameters
        outputs_1 = self.mlp_1(x)
        return outputs_1

# Sample data
input_size = 2
hidden_size = 5
output_size = 1
batch_size = 1
num_epochs = 2
learning_rate = 0.001

# Randomly generated dataset
x_train = torch.ones(2, input_size)
y_train = torch.ones(2, output_size)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize models, loss function, and optimizer
model_1 = MLP_1(input_size, hidden_size, output_size)
model_2 = MLP_2(input_size, hidden_size, output_size)
model_3 = MLP_3(model_1, model_2)
criterion = nn.MSELoss()
optimizer_2 = optim.Adam(model_2.parameters(), lr=learning_rate)

# Freeze the parameters of model_1
for param in model_1.parameters():
    param.requires_grad = False

print("\nInitial weights of model_3:")
for name, param in model_3.named_parameters():
    print(name, param.data)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        print(f'++++++++++++++++++epoch is {epoch}')
        print(f'------------------{inputs},|||||||||, {targets}')

        # Forward pass through model_3 which handles both model_1 and model_2
        outputs_3 = model_3(inputs)
        print(f'output of model 3 is: {outputs_3}')

        # Calculate loss
        loss = criterion(outputs_3, targets)
        print(f'loss is {loss}')

        # Ensure loss depends on model_2's parameters
        optimizer_2.zero_grad()
        loss.backward()

        for name, param in model_2.named_parameters():
            print(f'{name} grad before step: {param.grad}')

        # Backward pass and optimization for model_2
        optimizer_2.step()

        # Check parameters after optimizer step
        print('model 2 params after update are')
        with torch.no_grad():
            for param in model_2.parameters():
                print(param)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print()

print("Training complete.")

new_data = torch.tensor([1, 1], dtype=torch.float)
print(new_data)

predictions_3 = generate(model_3, new_data)

print("Generated predictions from model_3:", predictions_3)
