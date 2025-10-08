#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = {}
current_data = []
current_name = None

with open('abaqus_results.rpt', 'r') as f:
    for line in f:
        # Detect dataset header: line with at least two words, first is 'X'
        tokens = line.strip().split()
        if len(tokens) >= 2 and tokens[0] == 'X':
            # Save previous dataset
            if current_name and current_data:
                datasets[current_name] = np.array(current_data)
            # Start new dataset
            current_name = ' '.join(tokens)
            current_data = []
        elif line.strip() and not (line.strip().startswith('X') or line.strip() == ''):
            # Try to parse data lines
            try:
                values = [float(x.replace('E', 'e')) for x in line.split()]
                if len(values) == 2:
                    current_data.append(values)
            except Exception:
                pass  # skip lines that can't be parsed
    # Save last dataset
    if current_name and current_data:
        datasets[current_name] = np.array(current_data)
        
print(datasets.items())

# Plot all datasets
for name, data in datasets.items():
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], label=name)
    plt.xlabel(name.split()[0])
    plt.ylabel(name.split()[1] if len(name.split()) > 1 else '')
    plt.title(name)
    plt.legend()
    plt.savefig(str(name), dpi=400, bbox_inches='tight')
    plt.show()
    
    
# Meshsize = 0.01 (m)
# U2: -1.016 e-4
# Meshsize = 0.005
# -1.032 e-4
# Meshsize = 0.001
# -1.038 e-4
# Meshsize = 0.0005
# -1.038 e-4

# Data saved for this simulation

# L=3

# Meshsize = 0.01
# -1.148 e-1
# Meshsize = 0.005
# -1.165 e-1
# Meshsize = 0.001
# -1.170 e-1
# Meshsize = 0.0005
# -1.170 e-1

    
u2_3m = [-1.016e-4, -1.032e-4, -1.038e-4, -1.038e-4]
u2_03m = [-1.148e-1, -1.165e-1, -1.170e-1, -1.170e-1]
meshsize = [0.01, 0.005, 0.001, 0.0005]

name = 'Mesh convergence (3m)'
plt.figure()
plt.plot(meshsize, u2_3m,'X-')
plt.title(name)
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
plt.savefig(str(name), dpi=400, bbox_inches='tight')
plt.show()

name = 'Mesh convergence (03m)'
plt.figure()
plt.title(name)
plt.plot(meshsize, u2_03m,'X-')
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
plt.savefig(str(name), dpi=400, bbox_inches='tight')
plt.show()


datasets = {}
current_data = []
current_name = None

with open('abaqus_results_mises.rpt', 'r') as f:
    for line in f:
        # Detect dataset header: line with at least two words, first is 'X'
        tokens = line.strip().split()
        if len(tokens) >= 2 and tokens[0] == 'X':
            # Save previous dataset
            if current_name and current_data:
                datasets[current_name] = np.array(current_data)
            # Start new dataset
            current_name = ' '.join(tokens)
            current_data = []
        elif line.strip() and not (line.strip().startswith('X') or line.strip() == ''):
            # Try to parse data lines
            try:
                values = [float(x.replace('E', 'e')) for x in line.split()]
                if len(values) == 2:
                    current_data.append(values)
            except Exception:
                pass  # skip lines that can't be parsed
    # Save last dataset
    if current_name and current_data:
        datasets[current_name] = np.array(current_data)
        
print(datasets.items())

# Plot all datasets
for name, data in datasets.items():
    if name == 'X 3m':
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label= 'smises_bottom_X_03m')
        plt.xlabel(name.split()[0])
        plt.ylabel('smises_bottom_X_03m')
        plt.title('smises_bottom_X_03m')
        plt.legend()
        plt.savefig(str('smises_bottom_X_03m'), dpi=400, bbox_inches='tight')
        plt.show()
    if name == 'X m':
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label= 'smises_bottom_X_3m')
        plt.xlabel(name.split()[0])
        plt.ylabel('smises_bottom_X_3m')
        plt.title('smises_bottom_X_3m')
        plt.legend()
        plt.savefig(str('smises_bottom_X_3m'), dpi=400, bbox_inches='tight')
        plt.show()
    
#%%
