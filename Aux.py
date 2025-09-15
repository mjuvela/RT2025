from numpy import *
from numpy.random import rand
from matplotlib.pylab import *
import scipy.constants as C

c    =  C.c             # speed of light
h    =  C.h             # Planck constant
k    =  C.k             # Boltzmann constant
EPS  =  1.0e-6          # epsilon
PC   =  3.0856775e+16   # parsec


def Planck(f, T):
    return (2.0*h*f**3/c**2)  /  ( exp(h*f/(k*T)) - 1.0 )


def create_cloud(NX, NY, NZ):
    # Generate density distribution for the model
    # Peak density 1e4 cm-3  =  1e10 m-3
    I, J, K = indices((NX, NY, NZ), float32)
    I -= (NX-1)/2
    J -= (NY-1)/2
    K -= (NZ-1)/2
    n  = exp(-0.005*(I*I+J*J+K*K))
    n  = 1.0e10*asarray(n, float32)
    if (0):
        imshow(n[NX//2,:,:])
        show(block=True)
        sys.exit()
    return n


def initialise_background_package(NX, NY, NZ):
    # Generate photon package with random position and direction
    u       = rand()
    ct      = sqrt(rand())
    st      = sqrt(1.0-ct*ct)
    phi     = 2.0*pi*rand()
    cp      = cos(phi)
    sp      = sin(phi)
    ###
    if (u<1/6):   #  side X=0
        x, y, z =  EPS,       NY*rand(), NZ*rand()
        u, v, w =  ct,        st*cp,     st*sp
    elif (u<2/6): # side X=NX
        x, y, z =  NX-EPS,    NY*rand(), NZ*rand()
        u, v, w = -ct,        st*cp,     st*sp
    elif (u<3/6): # side Y=0
        x, y, z =  NX*rand(), EPS,       NZ*rand()
        u, v, w =  st*cp,     ct,        st*sp
    elif (u<4/6): # side Y=NY
        x, y, z =  NX*rand(), NY-EPS,    NZ*rand()
        u, v, w =  st*cp,     -ct,       st*sp
    elif (u<5/6): # side Z=0
        x, y, z =  NX*rand(), NY*rand(), EPS
        u, v, w =  st*cp,     st*sp,     ct
    else:         # side Z=NZ
        x, y, z =  NX*rand(), NY*rand(), NX-EPS
        u, v, w =  st*cp,     st*sp,     -ct

    return x, y, z, u, v, w




def initialise_background_package_2(NX, NY, NZ, pp):
    # Generate photon package, systematic selection of surface element,
    # random position on the element
    # ... 
    u       =  rand()
    ct      =  sqrt(rand())       # cos(theta), theta wrt surface normal
    st      =  sqrt(1.0-ct*ct)    # sin(theta)
    phi     =  2.0*pi*rand()      # phi rotation around the normal
    cp      =  cos(phi)
    sp      =  sin(phi)
    ###
    AREA    =  2*(NY*NZ+NX*NZ+NX*NY)  # number of surface elements
    k       =  pp % AREA              # index of the surface element
    # print("%6d  %6d" % (pp, k))
    if (k<(NY*NZ)):                     #  side X=0
        iy      =  k  % NY
        iz      =  k // NY
        x, y, z =  EPS,       iy+rand(), iz+rand()
        u, v, w =  ct,        st*cp,     st*sp
    else:
        k  -=  NY*NZ
        if (k<(NY*NZ)):                 #  side X=NX
            iy      =  k  % NY
            iz      =  k // NY
            x, y, z =  NX-EPS,    iy+rand(), iz+rand()
            u, v, w = -ct,        st*cp,     st*sp
        else:
            k -= NY*NZ
            if (k<NX*NZ):               #  side Y=0
                ix      =  k  % NX
                iz      =  k // NX
                x, y, z =  ix+rand(), EPS,       iz+rand()
                u, v, w =  st*cp,     ct,        st*sp
            else:
                k -= NX*NZ
                if (k<(NX*NZ)):         #  side Y=NY
                    ix      =  k  % NX
                    iz      =  k // NX
                    x, y, z =  ix+rand(), NY-EPS,    iz+rand()
                    u, v, w =  st*cp,     -ct,       st*sp
                else:
                    k -= NX*NZ
                    if (k<NX*NY):       #  side Z=0
                        ix      =  k  % NX
                        iy      =  k // NX
                        x, y, z =  ix+rand(), iy+rand(), EPS
                        u, v, w =  st*cp,     st*sp,     ct
                    else:               #  side Z=NZ
                        k      -=  NX*NY
                        ix      =  k  % NX
                        iy      =  k // NX
                        x, y, z =  ix+rand(), iy+rand(), NZ-EPS
                        u, v, w =  st*cp,     st*sp,     -ct

    return x, y, z, u, v, w, 1.0




def get_step_length(x, y, z,  u, v, w):
    # take step to next cell boundary, return the distance without updating (x,y,z)
    if (u>0.0):
        dx  =  (1.0+EPS-fmod(x, 1.0)) / u
    else:
        dx  =     -(EPS+fmod(x, 1.0)) / u
    if (v>0.0):
        dy  =  (1.0+EPS-fmod(y, 1.0)) / v
    else:
        dy  =     -(EPS+fmod(y, 1.0)) / v
    if (w>0.0):
        dz  =  (1.0+EPS-fmod(z, 1.0)) / w
    else:
        dz  =     -(EPS+fmod(z, 1.0)) / w
    return min([dx, dy, dz])



def get_cell_indices(x, y, z, NX, NY, NZ):
    # Return integer cell indices for position (x, y, z)
    if ((x<=0.0)|(x>=NX)): return -1, -1, -1
    if ((y<=0.0)|(y>=NY)): return -1, -1, -1
    if ((z<=0.0)|(z>=NZ)): return -1, -1, -1
    return  int(floor(x)), int(floor(y)), int(floor(z))



def read_dust_file(name):
    # Read dust optical properties from text file
    lines  =  open(name).readlines()
    A      =  float(lines[1].split()[0])       # dust-to-gas number ratio
    B      =  float(lines[2].split()[0])*0.01  # grain size [m]
    d      =  loadtxt(name, skiprows=4)
    freq   =  d[:,0]
    g      =  d[:,1]
    Kabs   =  A * pi*B**2 * d[:,2]  # cross section per H-atom [m2/H]
    Ksca   =  A * pi*B**2 * d[:,3]  # cross section per H-atom [m2/H]
    return freq, g, Kabs, Ksca



def isotropic_scattering(u, v, w):
    # just random direction uniformly over 4 pi solid angle
    cos_theta =  -1.0+2.0*rand()
    sin_theta =  sqrt(1.0-cos_theta**2)
    phi       =  2.0*pi*rand()
    u         =  sin_theta*cos(phi)
    v         =  sin_theta*sin(phi)
    w         =  cos_theta
    return u, v, w


# def HG_scattering(u, v, w, g):
#     cos_theta =  (1.0+g*g-((1-g*g)/(1-g+2*h*rand()))**2) / (2.0*g)
#     phi       =  2.0*pi*rand()
#     # new vector in coordinate system where +Z is current direction of propagation
#     sin_theta =  sqrt(1.0-cos_theta**2)
#     uu  =  sin_theta*cos(phi)
#     vv  =  sin_theta*sin(phi)
#     ww  =  cos_theta
#     # rotate current +Z to the direction (u,v,w)
#     theta     =  arccos(uu*u+vv*v+ww*w)   # dot product = a * b * cos(angle)
#     phi       =  arctan2(u, v)
#     # rotate angle theta around y, then angle phi around z
#     tmp = asarray([uu, vv, ww], flot32)
#     R1  = asarray([ [cos(theta),  0,  sin(theta)],
#                     [0,           1,  0         ],
#                     [-sin(theta), 0, cos(theta) ]])
#     R2  = asarray([ [cos(phi),  sin(phi),  0],
#                     [-sin(phi), cos(phi),  0],
#                     [0,         0,         1]])
#     tmp =  matmul(R2, matmul(r1, tmp))
#     return tmp[0], tmp[1], tmp[2]



def get_integration_weights(freq):
    """
    Return integration weights for trapezoidal integration over frequency axis,
    given the frequency grid freq.
    """
    N       =  len(freq)
    weights =  zeros(N, float32)
    for i in range(N):   #  loop over intervals
        if (i>0):        #  integral +=  0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
            weights[i] += 0.5*(freq[i]-freq[i-1])
        if (i<(N-1)):    #  integral +=  0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
            weights[i] += 0.5*(freq[i+1]-freq[i])
    return weights
    


def test_integration_weights():
    """
    Integrate y=x**2 over x=[0.5, 4.0]
    Y = x**3/3,  integral  (4.0**3-(0.5)**3)/3 = 21.291666667
    """
    I0 = 21.291666667
    for N in [ 4, 10, 100]:   # try different number of points
        x   =  logspace(log10(0.5), log10(4.0), N)
        w   =  get_integration_weights(x)
        I   =  sum(w*(x**2.0))
        print("points  %3d,  estimate %10.4e,  error %.3f per cent\n"
              % (N, I,  100.0*(I-I0)/I0))
        
