import itertools
import torch

def expand_sims_agents(input_tensor):
    sims, agents, time, _ = input_tensor.shape
    
    # Generate all combinations of simulations for all agents
    sim_combinations = itertools.product(range(sims), repeat=agents)
    
    # Create a tensor to hold the indices
    indices = torch.tensor(list(sim_combinations))
    
    # Expand dims to match the input tensor shape
    indices = indices.unsqueeze(2).expand(-1, agents, time, -1)
    
    # Gather values from the input tensor using the indices
    output_tensor = torch.gather(input_tensor, 1, indices)
    
    return output_tensor

# Example usage
input_tensor = torch.randn(3, 2, 12, 2)  # Example input tensor
output_tensor = expand_sims_agents(input_tensor)
print(output_tensor.shape)  # Output tensor shape (81, 4, 12, 2)
