#%%
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from pathlib import Path

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Definitioner, ignorera

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def new_task(string):
    print_string = '\n' + '=' * 80 + '\n' + '=' * 80 + '\n' + str(string) + '\n' + '=' * 80 + '\n' + '=' * 80 + '\n'
    return print(print_string)

def new_subtask(string):
    print_string = '\n' + '=' * 80 + '\n' + str(string) + '\n' + '=' * 80 + '\n'
    return print(print_string)

def printt(string):
    print()
    print('=' * 40)
    print(string)
    print('=' * 40)
    print()

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
dpi = 500

plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=BIGGER_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.rc('figure', figsize=(8,4))

script_dir = Path(__file__).parent

def sfig(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    print('figure name: ', fig_name)
    
def figgg(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    plt.show()
    print('figure name: ', fig_name)

#%%

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

new_task('Uppgift 1')

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ::::::::::::::::::::::::::::::::::::::::
# Definiera funktion
# ::::::::::::::::::::::::::::::::::::::::

def y(x):
    return x**2

# ::::::::::::::::::::::::::::::::::::::::
# Vektorer som ska plottas
# ::::::::::::::::::::::::::::::::::::::::

x_vals = np.linspace(0, 100, 20)
y_vals = y(x_vals)

# ::::::::::::::::::::::::::::::::::::::::
# Plotta vektorerna
# ::::::::::::::::::::::::::::::::::::::::

plt.figure()
plt.plot(x_vals, y_vals, color='blue', label='plot')
plt.scatter(x_vals, y_vals, color='red', label='scatter')
plt.title('titel')
plt.xlabel('x')
plt.ylabel('y')
figgg('figurnamn') # figurfunktion, ange endast namn på figuren, sparas en .png fil

# ::::::::::::::::::::::::::::::::::::::::
# Printa resultat
# ::::::::::::::::::::::::::::::::::::::::

print(f'värdet för x = 5 (avrundat till 2 decimaler): {y(x=5):.2f}')

# ::::::::::::::::::::::::::::::::::::::::
# For loop
# ::::::::::::::::::::::::::::::::::::::::

for i in range(len(x_vals)):
    x_val = x_vals[i]
    y_val = x_val**3
    
    # printa enbart var 5:e värde
    if i // 5 == 0:
        print(f'y_val (avrundat till 2 decimaler och tiopotenser) {y_val:.2e}')


#%%