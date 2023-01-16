#learn to use styling for matplotlib
import numpy as np
import matplotlib.pyplot as plt 



def testplot():
    x=np.linspace(0,6.6)
    y=np.sin(x)
    f,axs=plt.subplots(ncols=2,sharey=True)
    axs[0].plot(x,y,label='sin(x)')
    axs[0].set_ylabel('Y-axis')
    axs[0].set_xlabel('X-axis 1')
    axs[0].text(0,-3.5,"some text")
    axs[0].legend()
    axs[1].plot(x,x*y,label='x sin(x)')
    axs[1].set_xlabel('X-axis 1')
    axs[1].text(0,-3.5,"even more text")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('test.pdf')

testplot()