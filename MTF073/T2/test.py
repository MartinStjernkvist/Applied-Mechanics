for j in range(1, nJ - 1):
        

        
        for i in range(2, nI - 1):
            
            d = (aN[i, j] * phi[i, j + 1] + 
                 aS[i, j] * phi[i, j - 1] + 
                 Su[i, j])
            
            # Using P[i-1] (West)
            denominator = a[i, j] - c[i, j] * P[i - 1, j]
            
            P[i, j] = b[i, j] / denominator
            Q[i, j] = (d + c[i, j] * Q[i - 1, j]) / denominator
        
        for i in reversed(range(1, nI - 1)):
            phi[i, j] = P[i, j] * phi[i + 1, j] + Q[i, j]
    

    for i in range(1, nI - 1):

        a = aP
        b = aN
        c = aS
        
        j = 1
        
        d = (aE[i, j] * phi[i + 1, j] + 
             aW[i, j] * phi[i - 1, j] + 
             Su[i, j])
        
        P[i, j] = b[i, j] / a[i, j]
        Q[i, j] = (d + c[i, j] * phi[i, j - 1]) / a[i, j]
        
        for j in range(2, nJ - 1):
        
            d = (aE[i, j] * phi[i + 1, j] + 
                 aW[i, j] * phi[i - 1, j] + 
                 Su[i, j])
            
            denominator = a[i, j] - c[i, j] * P[i, j - 1]
            
            P[i, j] = b[i, j] / denominator
            Q[i, j] = (d + c[i, j] * Q[i, j - 1]) / denominator
        
        for j in reversed(range(1, nJ - 1)):
            phi[i, j] = P[i, j] * phi[i, j + 1] + Q[i, j]