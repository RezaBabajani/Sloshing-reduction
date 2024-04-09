import numpy as np
import cv2
import os
from live_processing import live_processing
from command_law import command_law
from ga_operations import roulette_wheel_selection, single_point_crossover, mutate

# Setup
your_folder = '/path/to/your/folder'  # Update this path accordingly
os.makedirs(your_folder, exist_ok=True)
new_sub_folder = os.path.join(your_folder, 'Initial population')
os.makedirs(new_sub_folder, exist_ok=True)

# Configuration for the production line and GA
T = 1  # Total time
N_Frame = 200
t = np.linspace(0, T, N_Frame)
xf = 0.09  # Movement distance of the bottle
N = 4  # Number of genes
npop = 16  # Population size
maxit = 100  # Max iterations
maxstall_gen = 2  # Stall generations
obj_tol = 1e-4  # Objective tolerance
Acceptable_obj_value = 1.6e-3  # Acceptable objective value

# Initialize webcam
vidObj = cv2.VideoCapture(0)  # Adjust the camera index
ret, img_test = vidObj.read()
if not ret:
    print("Failed to grab a frame from the camera.")
    exit()

# Assuming manual ROI selection; this could be automated as needed
cv2.imshow('Select ROI and press ENTER', img_test)
bbox = cv2.selectROI(img_test, False)
cv2.destroyAllWindows()
vidObj.release()  # Release the camera after setup

# Initialize population and evaluate
pop = np.zeros((npop, N + 1))  # Including cost
for i in range(npop):
    pop[i, :-1] = np.random.rand(N)  # Initialize with random values
    pop[i, -1] = live_processing(pop[i, :-1], bbox)  # Dummy arguments for live_processing

# Main GA loop
stall_gen = 0
gen = 1
while gen <= maxit:
    # Selection
    selected = roulette_wheel_selection(pop, npop)

    # Crossover
    offspring = np.empty((0, N + 1))
    for i in range(0, len(selected), 2):
        parent1, parent2 = selected[i], selected[i + 1]
        child1, child2 = single_point_crossover(parent1[:-1], parent2[:-1])
        offspring = np.vstack((offspring, np.zeros((2, N + 1))))
        offspring[-2, :-1], offspring[-1, :-1] = child1, child2
        offspring[-2, -1] = live_processing(child1, bbox)  # Dummy arguments for live_processing
        offspring[-1, -1] = live_processing(child2, bbox)

    # Mutation
    for i in range(len(offspring)):
        offspring[i, :-1] = mutate(offspring[i, :-1])
        offspring[i, -1] = live_processing(offspring[i, :-1], bbox)

    # Combine and select the best to form the new population
    combined = np.vstack((pop, offspring))
    combined = combined[combined[:, -1].argsort()]  # Sort by cost
    pop = combined[:npop, :]

    # Update and display the best cost
    print(f"Generation {gen}: Best Cost = {pop[0, -1]}")

    # Stopping criteria check
    if gen > 1 and np.abs(pop[0, -1] - best_cost) <= obj_tol and pop[0, -1] <= Acceptable_obj_value:
        stall_gen += 1
        if stall_gen >= maxstall_gen:
            print("Optimization terminated due to stall generations limit.")
            break
    else:
        stall_gen = 0
    best_cost = pop[0, -1]

    if gen == maxit:
        print("Optimization terminated due to maximum iterations.")
        break

    gen += 1

# Finalize
print("GA optimization completed.")
