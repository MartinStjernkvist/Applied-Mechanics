#%%
import re

latex_code = r"""
\begin{equation}
    \begin{split}
        p_\text{truss} = -0.00917333333333334\\
        p_\text{frame}^P = -0.00916602635302354\\
        p_\text{frame}^q = -0.00629280127047815
    \end{split}
\end{equation}
"""

def smart_latex_formatter(match):
    val = float(match.group(0))
    
    # 1. Handle absolute zero
    if val == 0:
        return "0"
        
    # 2. Get the exponent magnitude
    # We format to scientific string first to extract the true exponent easily
    sci_str = "{:.2e}".format(val) 
    base, exponent_str = sci_str.split('e')
    exp_int = int(exponent_str)
    
    # --- FAILSAFE CHECK ---
    # If the exponent is 0 (numbers 1-9) or 1 (numbers 10-99), 
    # return standard decimal formatting.
    # You can add -1 to this list if you also want 0.1-0.9 to stay decimal.
    if exp_int in [0, 1]:
        # returns simple float (e.g. "12.45")
        # .4g keeps significant digits but avoids unnecessary zeros
        return "{:.4g}".format(val) 
    
    # 3. Otherwise, return LaTeX Scientific Notation
    return r"{} \cdot 10^{{{}}}".format(base, exp_int)

# Regex matches floating point numbers
new_latex = re.sub(r'-?\d+\.\d+(?:e[+-]?\d+)?', smart_latex_formatter, latex_code)
print(new_latex)

for i in range(1,5):
    print(i)
#%%