import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
#test style on a figure
x=np.linspace(0,0.25*np.pi)
N=6
styles=plt.style.available
styles=['seaborn-colorblind','tableau-colorblind10','seaborn-poster']
for s in styles:
    with plt.style.context(s):
        for i in range(N):
            plt.plot(x,np.sin((i+1)*x),label=f'sin ({i+1}x)')
        plt.legend()
        plt.xlabel(r'$\theta \, [radians]$')
        plt.ylabel(r'$\sin (\theta)$')
        plt.show()