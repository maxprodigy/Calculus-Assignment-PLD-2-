

```markdown
# Dynamic Derivative Calculator and Function Visualization

This project involves calculating the derivative of a function, classifying critical points, plotting graphs, and finding the area under the curve using Python. The library `sympy` is utilized for symbolic mathematics, while `matplotlib` is used for plotting.

## Tasks Overview

### Task 1: Calculate Derivative
A dynamic function `dy_dx` is created to compute the derivative of any function \( f(x) \).

```python
import sympy as sp

# Create a function that calculates the derivative for any function
def dy_dx(function):
    # Define the variable x
    x = sp.symbols('x')

    # Convert the input function string into a sympy expression
    f_x = sp.sympify(function)

    # Compute the derivative of the function f(x) with respect to x
    derivative = sp.diff(f_x, x)

    # Return the derivative
    return derivative
```

### Task 2: Test the Derivative Function
The derivative function is tested with an equation that has at least two minima and two maxima.

```python
import sympy as sp

# Function to classify critical points of a given function.
def classify_critical_points(func):
    x = sp.symbols('x')  # Define a symbolic variable 'x'.
    f_x = sp.sympify(func)  # Convert function string to a sympy expression.

    first_derivative = sp.diff(f_x, x)  # Compute the first derivative.
    second_derivative = sp.diff(first_derivative, x)  # Compute the second derivative.

    critical_points = sp.solve(first_derivative, x)  # Find critical points.

    # Classify points based on the second derivative.
    min_points = [p.evalf() for p in critical_points if second_derivative.subs(x, p) > 0]
    max_points = [p.evalf() for p in critical_points if second_derivative.subs(x, p) < 0]

    return min_points, max_points, first_derivative

# Define the function to analyze.
function_string = "x**5 - 10*x**4 + 35*x**3 - 50*x**2 + 24*x"

# Call the function to classify critical points.
min_points, max_points, first_derivative = classify_critical_points(function_string)
print("First Derivative:", first_derivative)
```

### Task 3: Plot the Function
A plot of the function is generated, marking the identified minima and maxima.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Generate x values for plotting
x_values = np.linspace(-1, 6, 1000)
f_func = sp.lambdify(sp.symbols('x'), f_x)
y_values = f_func(x_values)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="f(x)")
# Mark minima and maxima points
```

### Task 4: Create Arrays of Minima and Maxima
Arrays are created to store the calculated minima and maxima.

```python
# Find the minima and maxima points
arr_minimas, arr_maximas = classify_critical_points(function_string)
print(f'The array of minimas is {arr_minimas}')
print(f'The array of maximas is {arr_maximas}')
```

### Task 5: Area Under the Curve
Calculating the area under the curve between local extrema using numerical integration.

```python
from scipy.integrate import quad

# Define the function to analyze
def f(x):
    return x**5 - 10*x**4 + 35*x**3 - 50*x**2 + 24*x

# Calculate areas under the curve
area1, _ = quad(f, local_max_1, local_min_1)  # Area between first max and first min
area2, _ = quad(f, local_min_1, local_max_2)  # Area between first min and second max
area3, _ = quad(f, local_max_2, local_min_2)  # Area between second max and second min

# Print the absolute areas calculated
print(f"Total area under the curve: {total_area:.4f}")
```

### Final Plot
The final plot visualizes the function along with the global minima and maxima.

```python
plt.figure(figsize=(12, 8))
plt.plot(x, y, label='f(x)', color='blue')
# Plot global minima and maxima
```

## Results
- **Global Minima:** (3.64, -3.63)
- **Global Maxima:** (0.36, 3.63)

## Requirements
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `sympy`, `scipy`

## How to Run
1. Install the required libraries using pip:
   ```bash
   pip install numpy matplotlib sympy scipy
   ```
2. Run the script in your preferred Python environment.

