import numpy as np
import matplotlib.pyplot as plt
import galsim #install with conda install -c conda_forge galsim

def half_mass_radius(Md,Rd,Mb,Rb,tol=1.e-6,figure=False):
    '''calculate half mass radius for disk(n=1)+bulge(n=4) galaxy
    using bisection'''
    disk_fraction=Md/(Md+Mb)
    bulge_fraction=Mb/(Md+Mb)
    if bulge_fraction==0:
        disk=galsim.Sersic(scale_radius=Rd,n=1)  
        return disk.half_light_radius     
    if disk_fraction==0:
        return Rb
    disk=galsim.Sersic(scale_radius=Rd,n=1)
    bulge=galsim.Sersic(half_light_radius=Rb,n=4)
    #starting points for bisection
    a=bulge.half_light_radius
    b=disk.half_light_radius
    if b < a:
        a=disk.half_light_radius
        b=bulge.half_light_radius

    #bisection
    Ma=disk_fraction*disk.calculateIntegratedFlux(a)+bulge_fraction*bulge.calculateIntegratedFlux(a)
    Mb=disk_fraction*disk.calculateIntegratedFlux(b)+bulge_fraction*bulge.calculateIntegratedFlux(b)
    if np.sign(Ma-0.5)==np.sign(Mb-0.5):
        raise Exception("a and b do not bound a root")
    while(b-a > tol):
        m=0.5*(a+b)
        f=(disk_fraction*disk.calculateIntegratedFlux(m)+
            bulge_fraction*bulge.calculateIntegratedFlux(m))-0.5
        if np.sign(f)==1:
            b=m
        else:
            a=m
    half_radius=0.5*(a+b)
    if figure:
        half=np.array([0.45,0.55]) #used for half marks
        R=np.arange(0,3*b,0.02)
        N=len(R)
        yd=np.zeros(N)
        yb=np.zeros(N)
        for i in range(N):
            yd[i]=disk_fraction*disk.calculateIntegratedFlux(R[i])
            yb[i]=bulge_fraction*bulge.calculateIntegratedFlux(R[i])
        plt.plot(R,yd,color='b',label='disk')
        plt.plot(R,yb,color='r',label='bulge')
        plt.plot(R,yd+yb,color='k',label='total')
        plt.plot([bulge.half_light_radius,bulge.half_light_radius],bulge_fraction*half,color='g')
        plt.plot([disk.half_light_radius,disk.half_light_radius],disk_fraction*half,color='g')
        plt.plot([half_radius,half_radius],half,color='g')
        plt.xlim([0,3*b])
        plt.ylim([0,1.0])
        plt.legend()
        plt.show()
    return half_radius

if __name__=='__main__':
    r_disk=1.109940e+00
    r_bulge=3.582890e-04
    mbulge=9.893530e+04
    mstar=5.311550e+05
    print(half_mass_radius(mstar-mbulge,r_disk,mbulge,0.1,figure=True))