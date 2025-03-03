# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:42:11 2025

@author: aiden
"""

#Bioinformatics example >>>
#
#   Task: Given the following DNA sequences, compute a similarity score between every pair of sequences. The similarity score will simply be the count of identical nucleotides at corresponding positions.
#
#   Plot the similarity scores using a Seaborn heatmap function:
#
#import seaborn as sns
#sns.heatmap(name_of_your_array, annot=True, cmap='jet', xticklabels=sequences, yticklabels=sequences)

sequences = [  
    "ATCGTACG", 
    "ATCGTACC",
    "GTCAATCG",
    "ATGGTACT",
    "ATCGTACG",
    "GTCACTGG" 
]

#   Tip 1: test how this code works, and consider how you may use it
#
#   score = 0
#   for a, b in zip(sequences[0], sequences[1]):
#       score = score + (a == b)
#
#   note: for the adventurous, the above code can be re-written as: 
#   score = sum(a == b for a, b in zip(sequences[i], sequences[j]))
#
#
#   Tip 2: in this rare instance, a triple (!) nested for loop may be necessary.
#
#   //////////////////////////////////////////////


import numpy as np # imports numpy
import matplotlib.pyplot as plt # imports matplot
import seaborn as sns #imports seaborn

# Number of sequences
num_sequences = len(sequences) #gives length of sequences

# Matrix to store similarity scores
similarity_matrix = np.zeros((num_sequences, num_sequences)) #preallocate space for similarity values

# Nested loops to calculate pairwise similarity scores
for i in range(num_sequences):  # Outer loop for first sequence
    for j in range(num_sequences):  # Inner loop for second sequence
        score = sum(a == b for a, b in zip(sequences[i], sequences[j]))  # Count matching nucleotides
        similarity_matrix[i, j] = score  # Store the similarity score in the matrix

# Plot the similarity matrix as a heatmap
plt.figure(figsize=(10, 8)) # creates figure of size 10,8
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=sequences, yticklabels=sequences) # creates heatmap
plt.axis('equal') # sets axis sizes equal to one another
plt.title('Similarity Heatmap of DNA Sequences') # title for whole plot
plt.xlabel('Sequences') # title for x-axis
plt.ylabel('Sequences') # title for y-axis
plt.show() # shows the plot


# My recreation
sequences = [  
    "ATCGTACG", 
    "ATCGTACC",
    "GTCAATCG",
    "ATGGTACT",
    "ATCGTACG",
    "GTCACTGG" ]
num_sequences = len(sequences)
similarity_mat = np.zeros((num_sequences,num_sequences))
    
for i in range(num_sequences):
    for j in range(num_sequences):
        score = 0
        for k in range(len(sequences[1])): # all same length sequences or I'd use i as indexing veriable
            if sequences[i][k] == sequences[j][k]:  
                score += 1
                similarity_mat[i, j] = score
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=sequences, yticklabels=sequences)
    plt.axis('equal')
    plt.title('Similarity Heatmap of DNA Sequences') 
    plt.xlabel('Sequences')
    plt.ylabel('Sequences') 
    plt.show()
# Problem #2 Random Walk
import random
import matplotlib.pyplot as plt

iterations = 5000
count = [0] * (iterations)
for t in range(iterations):
    if random.random() >= 0.5:
       count[t] = count[t-1] + 1
    else:
       count[t] = count[t-1] - 1
plt.plot(range(iterations),count)
plt.show()

# Question 3
import matplotlib.pyplot as plt # imports matplot
plt.style.use('seaborn-v0_8-deep') #sets plotting style
import numpy as np # imports numpy

def f(x, y): # defining function f which consists of inputs x and y
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x) # outputs this operation performed on the input
    
x = np.linspace(0, 5, 500) # 0 to 5 with 500 points
y = np.linspace(0, 5, 400) # 0 to 5 with 400 points

X, Y = np.meshgrid(x, y) # all possible combos of x and y
Z = f(X, Y) # uses function f to output X and Y which are just x,y with the operations from line 86

contours = plt.contour(X, Y, Z, 3, colors='k') # plots contour lines
plt.clabel(contours, inline=True, fontsize=8) # labels colorbar

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5) #heatmap
plt.colorbar() # shows colorbar

# Question 3b
import matplotlib.pyplot as plt #import matplot
plt.style.use('seaborn-v0_8-deep') # chooses style
import numpy as np # import numpy
from scipy.stats import gaussian_kde # importing gaussian density function from scipt

mean = [0, 0] # choosing mean of gaussian
cov = [[1, 1], [1, 2]] # chooses covariance
x, y = np.random.multivariate_normal(mean, cov, 10000).T # generates gaussian data with mean and covariance as defined with 10,000 points transposed

data = np.vstack([x, y]) # stacks x and y on top of one another
kde = gaussian_kde(data) # performas kernel density estimate using previously imported function on our stacked x and y

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 200) # -3.5 to 3.5 with 200 points
ygrid = np.linspace(-6, 6, 200) # -6 to 6 with 200 points
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid) # all possible combinations of xgrid and ygrid
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()])) # evaluates kernel density estimate of a flattened and stacked x and y grid

# plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape), origin='lower',
           aspect='auto', extent=[-3.5, 3.5, -6, 6],
           cmap='Blues') # generates plot
plt.colorbar(label='density') #labels colorbar
plt.scatter(x, y, s=1, c='r', alpha=0.05) #scatters points

plt.xlim(-3.5, 3.5) # sets axis limits
plt.ylim(-6, 6) # sets axis limits