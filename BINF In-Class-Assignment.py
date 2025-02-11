# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:32:48 2025

@author: aiden
"""
import numpy as np
import matplotlib.pyplot as plt

names = ['Curie','Darwin','Turing']
birth_yrs = np.array([1867,1809,1912])
for i in range(len(names)):
    print(names[i],birth_yrs[i])
    
# Question 2
t = np.linspace(-1, 1, 200)
coefs = [2, 4, 3, 1.5, -0.5]
y = np.zeros(len(t))
for i in range(len(coefs)):
    y += coefs[i] * t**i
plt.plot(t,y)
plt.xlabel('t value')
plt.ylabel('y value')
plt.show()

# Question 3
# plt.style.available
plt.style.use('seaborn-v0_8-darkgrid') 

# Parameters
r = 0.15  # prey birthrate
a = 0.03 # prey catch rate of predators
b = 0.005 # converting caught prey into predator offspring efficiency
d = 0.1 # death rate of predators 
k = 200 # prey carrying capacity

# Initial conditions
prey_init = 40  # Initial number of prey
pred_init = 10  # Initial number of predators

time_steps = 5000  # Number of time steps to simulate

#   Tip 1: set up your code with the following structure:
#   - initalize zero arrays to store population numbers over time
#   - populate the above arrays at time=0 with the initial conditions
#   - loop over time_steps, and update the arrays at each time step
#
#
#   Tip 2: in your code, instead of relating P(t+1) to P(t), relate P(t) to P(t-1)


# Arrays to store population numbers over time
prey_pop = np.zeros(time_steps) # initalizes an array of 0's the length of the time
pred_pop = np.zeros(time_steps) # initalizes an array of 0's the length of the time

# Set initial population values
prey_pop[0] = prey_init # creates a variable that will be iteratively changed in the loop
pred_pop[0] = pred_init # creates a variable that will be iteratively changed in the loop

# For loop to simulate population changes over time
for t in range(1, time_steps):  # Outer loop over time steps (from 1 to the end of time_steps)
    # Update prey population (influenced by birthrate, carrying capacity, how often they are caught by predators and overall predator population)
    prey_pop[t] = prey_pop[t-1] + r * prey_pop[t-1] * (1 - prey_pop[t-1]/k) - a * prey_pop[t-1] * pred_pop[t-1]
    
    # Update predator population (influenced by efficiency of turning caught prey into new predators and death rate)
    pred_pop[t] = pred_pop[t-1] + b * prey_pop[t-1] * pred_pop[t-1] - d * pred_pop[t-1]


# Plot the results
plt.figure(figsize=(10, 6)) # figure 10,6 large
plt.plot(prey_pop, label='Prey Population', color='green') #plotting prey population in green and labeling the legend
plt.plot(pred_pop, label='Predator Population', color='red') # plotting predator population in red and labeling the legend
plt.title('Predator-Prey Population Dynamics') # title of the figure
plt.xlabel('Time Steps') # x-axis label
plt.ylabel('Population Size') # y-axis label
plt.legend() # requriing a legend
plt.show() # displaying the created plot