"""
given m and n, this script will show rotation angle, unit cell size
"""
import numpy as np
import math

def twisted_angle(m, n):
    return np.arccos(0.5*(m**2+4*m*n+n**2)/(n**2+m*n+m**2))*180/np.pi


def main():
    theta_cut = [20, 40]
    size_max = 50
    Nmax = 500
    
    
    mns = []
    thetas = []
    sizes = []
    for n in range(1, Nmax):
        for m in range(1,n+1):
            if math.gcd(m, n) !=1:
                continue
            thetas.append(twisted_angle(n,m))
            mns.append((m,n))
            sizes.append(4*(m**2+n**2+m*n))
    mns = np.array(mns)
    thetas = np.array(thetas)
    sizes = np.array(sizes)
    inds = np.argsort(thetas)
    
    thetas = thetas[inds]
    mns = mns[inds]
    sizes = sizes[inds]
    
    inds = np.intersect1d(np.where(thetas<=theta_cut[1])[0], np.where(thetas>=theta_cut[0])[0])
    inds = np.intersect1d(inds, np.where(sizes<=size_max)[0])
    for i in inds:
        print( thetas[i], mns[i], sizes[i])

if __name__=='__main__':
    main()
