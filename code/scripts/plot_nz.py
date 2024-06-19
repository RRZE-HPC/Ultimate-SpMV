import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def save_matrix_market_histogram(file_path, output_path):
    # Read the matrix market file
    matrix = scipy.io.mmread(file_path)

    # Extract the non-zero values
    if scipy.sparse.issparse(matrix):
        values = matrix.data
    else:
        values = matrix.flatten()

    # Plot the histogram
    plt.hist(values, bins=50, edgecolor='black')
    plt.title('Histogram of Matrix Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(output_path, dpi=300)
    plt.close()

# Example usage
matrix_name = "Fault_639"
file_path = '/home/vault/k107ce/k107ce17/bench_matrices/' + matrix_name + '.mtx'
output_path = matrix_name + '_values_hist.png'
save_matrix_market_histogram(file_path, output_path)