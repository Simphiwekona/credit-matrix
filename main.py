import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# Define credit ratings and transition labels
credit_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

# Observed transition matrix (example)
observed_transition_matrix = np.array([
    [0.90, 0.05, 0.03, 0.01, 0.005, 0.005, 0.005, 0],
    [0.02, 0.85, 0.1, 0.02, 0.005, 0.005, 0.005, 0],
    [0.01, 0.02, 0.80, 0.05, 0.03, 0.02, 0.03, 0],
    [0.005, 0.005, 0.02, 0.85, 0.05, 0.03, 0.01, 0],
    [0, 0.005, 0.005, 0.02, 0.85, 0.05, 0.03, 0],
    [0, 0, 0.01, 0.03, 0.05, 0.80, 0.10, 0.01],
    [0, 0, 0, 0.01, 0.03, 0.05, 0.85, 0.06],
    [0, 0, 0, 0, 0, 0, 0, 1]
])


# Define fitted transition matrix (example)
def fitted_transition_matrix(params):
    return np.array([
        [0.92, 0.03, 0.03, 0.01, 0.005, 0.005, 0.005, 0],
        [0.02, 0.87, 0.08, 0.02, 0.005, 0.005, 0.005, 0],
        [0.01, 0.02, 0.82, 0.04, 0.03, 0.02, 0.03, 0],
        [0.005, 0.005, 0.02, 0.87, 0.04, 0.03, 0.01, 0],
        [0, 0.005, 0.005, 0.02, 0.87, 0.04, 0.03, 0],
        [0, 0, 0.01, 0.03, 0.04, 0.80, 0.10, 0.02],
        [0, 0, 0, 0.01, 0.03, 0.05, 0.87, 0.04],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ]) + params[0]  # Parameter for Z adjustment


import tkinter as tk
import numpy as np

class TransactionMatrixApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Estimation")

        # Initialize and place widgets for parameter input
        self.params_label = tk.Label(root, text="Z Adjustment Parameter:")
        self.params_entry = tk.Entry(root)
        self.params_label.grid(row=0, column=0, padx=10, pady=10)
        self.params_entry.grid(row=0, column=1, padx=10, pady=0)

        # Button for matrix estimation
        self.estimate_button = tk.Button(root, text="Estimate", command=self.estimate)
        self.estimate_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Displaying results
        self.result_text = tk.Text(root, height=10, width=60)
        self.result_text.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def estimate(self):
        z_param = float(self.params_entry.get())

        # Example observed_transition_matrix, replace it with your data
        observed_transition_matrix = np.array([[1, 2], [3, 4]])

        estimate_matrix = self.fitted_transition_matrix(observed_transition_matrix, z_param)

        # Displaying result in the widget
        result_str = f"Z adjustment Parameter: {z_param}\n\n"
        result_str += "Observed Transaction Matrix:\n" + str(observed_transition_matrix) + "\n\n"
        result_str += "Estimated Transaction Matrix: \n" + str(estimate_matrix)

        self.result_text.delete(1.0, tk.END)  # Clearing the previous values
        self.result_text.insert(tk.END, result_str)

    def fitted_transition_matrix(self, observed_transition_matrix, z_param):
        estimate_matrix = observed_transition_matrix.astype(float)  # Convert to float
        estimate_matrix[:-1, :-1] += z_param
        
        return estimate_matrix


if __name__ == "__main__":
    root = tk.Tk()
    app = TransactionMatrixApp(root)
    root.mainloop()


# Objective function to minimize the difference between observed and fitted matrices
def objective(params):
    return np.sum((observed_transition_matrix - fitted_transition_matrix(params)) ** 2)


# Minimize the objective function to estimate Z
result = minimize(objective, [0], bounds=[(-0.5, 0.5)])  # Adjust the bounds as needed

# Calculate the estimated Z value
estimated_Z = result.x[0]

# Visualize the observed and fitted transition matrices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(observed_transition_matrix, cmap='YlGnBu', interpolation='nearest')
plt.colorbar()
plt.title("Observed Transition Matrix")
plt.xticks(np.arange(len(credit_ratings)), credit_ratings, rotation=45)
plt.yticks(np.arange(len(credit_ratings)), credit_ratings)

plt.subplot(1, 2, 2)
fitted_matrix = fitted_transition_matrix(result.x)
plt.imshow(fitted_matrix, cmap='YlGnBu', interpolation='nearest')
plt.colorbar()
plt.title(f"Fitted Transition Matrix (Z = {estimated_Z:.2f})")
plt.xticks(np.arange(len(credit_ratings)), credit_ratings, rotation=45)
plt.yticks(np.arange(len(credit_ratings)), credit_ratings)

plt.tight_layout()
plt.show()

print(f"Estimated Z value: {estimated_Z:.2f}")
