### EX7 Implementation of Link Analysis using HITS Algorithm
### DATE: 24-10-25
### AIM: To implement Link Analysis using HITS Algorithm in Python.
### Description:
<div align = "justify">
The HITS (Hyperlink-Induced Topic Search) algorithm is a link analysis algorithm used to rank web pages. It identifies authority and hub pages 
in a network of web pages based on the structure of the links between them.

### Procedure:
1. ***Initialization:***
    <p>    a) Start with an initial set of authority and hub scores for each page.
    <p>    b) Typically, initial scores are set to 1 or some random values.
  
2. ***Construction of the Adjacency Matrix:***
    <p>    a) The web graph is represented as an adjacency matrix where each row and column correspond to a web page, and the matrix elements denote the presence or absence of links between pages.
    <p>    b) If page A has a link to page B, the corresponding element in the adjacency matrix is set to 1; otherwise, it's set to 0.

3. ***Iterative Updates:***
    <p>    a) Update the authority scores based on the hub scores of pages pointing to them and update the hub scores based on the authority scores of pages they point to.
    <p>    b) Calculate authority scores as the sum of hub scores of pages pointing to the given page.
    <p>    c) Calculate hub scores as the sum of authority scores of pages that the given page points to.

4. ***Normalization:***
    <p>    a) Normalize authority and hub scores to prevent them from becoming too large or small.
    <p>    b) Normalize by dividing by their Euclidean norms (L2-norm).

5. ***Convergence Check:***
    <p>    a) Check for convergence by measuring the change in authority and hub scores between iterations.
    <p>    b) If the change falls below a predefined threshold or the maximum number of iterations is reached, the algorithm stops.

6. ***Visualization:***
    <p>    Visualize using bar chart to represent authority and hub scores.

### Program:

```python
import numpy as np
import matplotlib.pyplot as plt

def hits_algorithm(adjacency_matrix, max_iterations=100, tol=1.0e-6):
    num_nodes = len(adjacency_matrix)
    authority_scores = np.ones(num_nodes)
    hub_scores = np.ones(num_nodes)
    iteration_count = 0

    for i in range(max_iterations):
        iteration_count += 1

        # Authority update 
        new_authority_scores = np.dot(adjacency_matrix.T, hub_scores)
        total_auth = np.sum(new_authority_scores)
        if total_auth != 0:
            new_authority_scores /= total_auth 

        # Hub update 
        new_hub_scores = np.dot(adjacency_matrix, new_authority_scores)
        total_hub = np.sum(new_hub_scores)
        if total_hub != 0:
            new_hub_scores /= total_hub 

        # Check convergence
        authority_diff = np.linalg.norm(new_authority_scores - authority_scores, ord=2)
        hub_diff = np.linalg.norm(new_hub_scores - hub_scores, ord=2)

        if authority_diff < tol and hub_diff < tol:
            break

        authority_scores = new_authority_scores
        hub_scores = new_hub_scores

    return authority_scores, hub_scores, iteration_count


# Example adjacency matrix
adj_matrix = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0]
])

# Run HITS algorithm
authority, hub, iterations = hits_algorithm(adj_matrix)

print(f"\nAlgorithm converged in {iterations} iterations.\n")

# Display node-wise scores
for i in range(len(authority)):
    print(f"Node {i}: Authority Score = {authority[i]:.4f}, Hub Score = {hub[i]:.4f}")

# Rank Authority and Hub Scores separately
authority_rank = np.argsort(-authority)
hub_rank = np.argsort(-hub)

print("\nAuthority Rank:")
for rank, node in enumerate(authority_rank, start=1):
    print(f"Rank {rank}: Node {node} (Score = {authority[node]:.4f})")

print("\nHub Rank:")
for rank, node in enumerate(hub_rank, start=1):
    print(f"Rank {rank}: Node {node} (Score = {hub[node]:.4f})")

# Bar chart of authority vs hub scores
nodes = np.arange(len(authority))
bar_width = 0.35
plt.figure(figsize=(8, 6))
plt.bar(nodes - bar_width/2, authority, bar_width, label='Authority', color='blue')
plt.bar(nodes + bar_width/2, hub, bar_width, label='Hub', color='green')
plt.xlabel('Node')
plt.ylabel('Scores')
plt.title(f'Authority and Hub Scores (Converged in {iterations} Iterations)')
plt.xticks(nodes, [f'Node {i}' for i in nodes])
plt.legend()
plt.tight_layout()
plt.show()
```

### Output:
<img width="773" height="370" alt="Screenshot 2025-10-24 110625" src="https://github.com/user-attachments/assets/8cab8ae8-7bcf-4c0f-8575-6fb0926c488c" />

<img width="1068" height="672" alt="Screenshot 2025-10-24 110637" src="https://github.com/user-attachments/assets/101b87ec-4ca5-45fd-b820-0ebf4f4cec8c" />


### Result:
thus the program executed successfully
