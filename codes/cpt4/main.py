import torch
import torch.nn as nn

input = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3)
linear = nn.Linear(3,3)
# set the weights to 1.
linear.weight.data.fill_(1)
# set the bias to 0.
linear.bias.data.fill_(0)

# print the weights and outputs
print('weights: ', linear.weight)
print('bias: ', linear.bias)

# set learning rate to be 0.1, error is 1,
# and then back popogate the error to update the weights4

errors = torch.tensor([[1, 2, 3]], dtype=torch.float32)

learning_rate = 0.1
error = 1
linear.zero_grad()
output = linear(input)
u = torch.matmul(output, errors.T)

print('error: ', error)
print('u: ', u)

optimizer = torch.optim.SGD(linear.parameters(), lr=learning_rate)
u.backward(torch.tensor([[error]], dtype=torch.float32))
print("====compatarion====")
print('weights before: ', linear.weight)
print('bias before: ', linear.bias)
optimizer.step()
print('weights after: ', linear.weight)
print('bias after: ', linear.bias)