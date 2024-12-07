import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
from helper_functions import lr_schedule
from model.model import Chefformer

# Define the optimizer and scheduler
model = Chefformer()
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=0.00005, 
                              weight_decay=0.0005)

# Define the scheduler using your lr_schedule function (this should already be defined)
scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

# List to store learning rates
lr_history = []

# Number of steps to simulate (this could be the same number of steps you'd use in training)
num_steps = 7000  # You can adjust this based on your training loop

# Simulate the learning rate change before training starts
for step in range(num_steps):
    # Update the scheduler to get the new learning rate
    optimizer.step()
    scheduler.step()
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    #current_lr = scheduler.get_last_lr()[0]
    
    # Store it in the history list
    lr_history.append(current_lr)

    # Training step
    optimizer.zero_grad()
    #loss.backward()
    

for i, lr in enumerate(lr_history):
    if i % 100 == 0: print(f'step: {i}, lr: {lr}')

# Plot the learning rate curve
plt.plot(range(num_steps), lr_history)
plt.xlabel('Training Iters')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule over Training Iters')
plt.show()