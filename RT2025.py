from numpy import *
from numpy.random import rand
from matplotlib.pylab import *
import scipy.constants as C
from scipy.interpolate import interp1d

# Define constants
c    =  C.c             # speed of light
h    =  C.h             # Planck constant
k    =  C.k             # Boltzmann constant
EPS  =  1.0e-6          # epsilon
PC   =  3.0856775e+16   # parsec

# 
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

        
#################################################################################
#################################################################################
#################################################################################
        

NX, NY, NZ  =  32, 32, 32     # model dimensions, for coordinates 0.0<x<NX
GL          =  0.1*PC         # size of a single cell (0.1
NPP         =  10000          # number of photon packages (PP)
n           =  create_cloud(NX, NY, NZ)

# Read dust properties, Kabs in units of [m2/H]
freq, g, Kabs0, Ksca0 = read_dust_file("COM.dust")
NF = len(freq)   # number of frequencies

# Make Kabs and Ksca optical depths for unit density n=1 and distance s=GL
Kabs  = Kabs0 * GL
Ksca  = Ksca0 * GL
    
# Select one frequency (the one closest to V band) for single-frequency runs
ifreq =  argmin(abs(freq-c/0.55e-6))
f     =  freq[ifreq]

# Calculate number of photons entering the model in one second, divide by NPP
#   => number of photons per single photon package
Ibg     =  1.0e-13*Planck(f, 10000.0)  # diluted T=10000K black body as background
phot_bg =  (Ibg*pi*2*(NX*NY+NX*NZ+NY*NZ)*GL*GL) / (h*f) / NPP


if (0):
    # Test initialise_background_package() routine
    POS = zeros((1000, 3), float32)
    DIR = zeros((1000, 3), float32)
    for i in range(1000):
        x, y, z, u, v, w   =  initialise_background_package(NX, NY, NZ)
        photons            =  phot_bg
        POS[i,:] = [x,y,z]
        DIR[i,:] = [u,v,w]

    figure(1, figsize=(10,6))
    subplot(231)
    plot(POS[:,0], POS[:,1], 'r.')
    subplot(232)
    plot(POS[:,0], POS[:,2], 'r.')
    subplot(233)
    plot(POS[:,1], POS[:,2], 'r.')

    subplot(234)
    plot(DIR[:,0], DIR[:,1], 'r.')
    subplot(235)
    plot(DIR[:,0], DIR[:,2], 'r.')
    subplot(236)
    plot(DIR[:,1], DIR[:,2], 'r.')

    show(block=True)
    

if (0):
    # Test if stepping works => trace 10 PPs
    figure(1, figsize=(13, 4))
    ax1 = subplot(131)
    xlabel("X"), ylabel("Y")
    ax2 = subplot(132)
    xlabel("Z"), ylabel("Z")
    ax3 = subplot(133)
    xlabel("Y"), ylabel("Z")
    axis0 = [ -0.5, NX+0.5, -0.5, NX+0.5 ]
    ##
    for ipacket in range(10):  # loop over 10 photon packages
        color = [ 'k', 'b', 'g', 'r', 'c', 'm' ][ipacket % 6]
        x, y, z, u, v, w  = initialise_background_package(NX, NY, NZ)
        photons           = phot_bg
        i, j, k  = get_cell_indices(x, y, z, NX, NY, NZ)
        while(i>=0):
            s  =  get_step_length(x, y, z, u, v, w)
            x +=  s*u
            y +=  s*v
            z +=  s*w
            i, j, k = get_cell_indices(x, y, z, NX, NY, NZ)
            # plotting .. one point at a time
            axes(ax1)  # (X,Y)
            plot(x, y, '.', color=color)
            axis(axis0)
            axes(ax2)  # (X,Z)
            plot(x, z, '.', color=color)
            axis(axis0)
            axes(ax3)
            plot(y, z, '.', color=color)
            axis(axis0)
            draw()
            pause(0.02)
    show(block=True)
    sys.exit()



################################################################################



def simulate_single_frequency():
    # Simulation of a single frequency, store and plot the absorbed energy
    NPP = 40000
    ABS = zeros((NX, NY, NZ), float32)  # array for absorbed energy
    t0  = time.time()
    for ipacket in range(NPP):
        if (ipacket%1000==0):
            print("packet  %6d  --- %5.2f per cent" % (ipacket, 100.0*ipacket/NPP))
        x, y, z, u, v, w  =  initialise_background_package(NX, NY, NZ)
        photons           =  phot_bg
        i, j, k           =  get_cell_indices(x, y, z, NX, NY, NZ)
        while(i>=0):   # while PP remains inside the model volume
            s             =  get_step_length(x, y, z, u, v, w)  # step
            tau           =  s*Kabs[ifreq]*n[i,j,k]             # tau =  s * Kabs * n
            ABS[i,j,k]   +=  photons*(1.0-exp(-tau))            # absorbed part
            photons      *=  exp(-tau)                          # remaining part
            x            +=  s*u                                # coordinate update
            y            +=  s*v
            z            +=  s*w
            i, j, k       =  get_cell_indices(x, y, z, NX, NY, NZ) # index update
    print("Run time %.2f seconds" % (time.time()-t0))
    
    # show cross section of the absorbed energy
    close(1)
    figure(1, figsize=(13, 3.5))
    subplots_adjust(left=0.05, right=0.98, bottom=0.09, top=0.91, wspace=0.2)
    for i in range(3):
        ax = subplot(1, 3, 1+i)
        ix = [2, 7, NX//2][i]
        imshow(ABS[ix,:,:], cmap=cm.gist_stern)
        text(0.08, 0.08, 'x=%d' % ix, transform=ax.transAxes, size=16, backgroundcolor='w')
        colorbar()
    show(block=True)
    

def simulate_single_frequency_2():
    # Simulation of a single frequency, store and plot the absorbed energy
    # Includes isotropic scattering and Russian rouletter.
    NPP = 10000
    ABS = zeros((NX, NY, NZ), float32)  # array for absorbed energy
    t0  = time.time()
    for pp in range(NPP):
        if (pp%1000==0): print("pp  %6d --- %5.2f per cent" % (pp, 100.0*pp/NPP))
        x, y, z, u, v, w  =  initialise_background_package(NX, NY, NZ)
        photons           =  phot_bg   # initial number of photons in package
        i, j, k           =  get_cell_indices(x, y, z, NX, NY, NZ)
        tau0              =  -log(rand())  # free path for scattering
        while(i>=0):   # while PP remains inside the model volume
            s             =  get_step_length(x, y, z, u, v, w)  # step
            dtaus         =  s*Ksca[ifreq]*n[i,j,k]
            if (dtaus>tau0): # --- partial step and scattering ---
                # distance to in tau to location of scattering
                s           =  tau0/(Ksca[ifreq]*n[i,j,k])  # distance to point of scattering
                tau         =  s*Kabs[ifreq]*n[i,j,k]       # tau(abs.) for this step
                ABS[i,j,k] +=  photons*(1.0-exp(-tau))
                photons    *=  exp(-tau)                    # absorptions on the shorter step
                u, v, w     =  isotropic_scattering(u, v, w) # all photons to new a direction
                tau0        =  -log(rand())                 # new free path
            else:            # --- normal full step ---
                tau         =  s*Kabs[ifreq]*n[i,j,k]       # tau =  s * Kabs * n
                ABS[i,j,k] +=  photons*(1.0-exp(-tau))      # absorbed part
                photons    *=  exp(-tau)                    # remaining part
                tau0       -=  dtaus                        # tau(sca) for the full step
            x         +=  s*u                               # coordinate update
            y         +=  s*v
            z         +=  s*w
            i, j, k    =  get_cell_indices(x, y, z, NX, NY, NZ) # index update
            if (1): # Russian roulette
                if (photons<1.0e-3*phot_bg):  #   true == 1  =  0.5*0 + 0.5*2.0
                    if (rand()<0.5):
                        break
                    else:
                        photons *= 2.0
                
    # convert absorbed photons to absorbed energy per cell (and per 1 Hz)
    # normalised by density
    ABS *=  h*f / n

    print("--------------------------------------------------------------------------------")
    print("Optical depth %.3e" % (NX*Kabs[ifreq]*mean(ravel(n))))
    print("Run time %.2f seconds" % (time.time()-t0))
    print("Model size %.2f pc,  mean density %.3e H/m3" % (NX*GL/PC, mean(ravel(n))))
    print("Model optical depth  tau_abs = %.3e" % (mean(ravel(n))*Kabs[ifreq]*NX))
    print("--------------------------------------------------------------------------------")
    
    # show cross section of the absorbed energy
    close(1)
    figure(1, figsize=(13, 3.5))
    subplots_adjust(left=0.05, right=0.98, bottom=0.09, top=0.91, wspace=0.2)
    for i in range(3):
        ax = subplot(1, 3, 1+i)
        ix = [2, 7, NX//2][i]
        imshow(ABS[ix,:,:], cmap=cm.gist_stern, vmin=0, vmax=170.0)
        text(0.08, 0.08, 'x=%d' % ix, transform=ax.transAxes, size=16, backgroundcolor='w')
        colorbar()
    show(block=True)
    sys.exit()
    
    
    
def simulate_all_frequencies():
    # Simulation of all frequencies, calculate integrated energy and save to file
    SUMABS  =  zeros((NX, NY, NZ), float32)  # total absorbed energy
    ABS     =  zeros((NX, NY, NZ), float32)  # absorbed photons per 1 Hz band
    weights =  get_integration_weights(freq)

    NPP     =  1000
    t0      =  time.time()
    for ifreq in range(NF):
        print("Frequency %3d / %3d" % (ifreq+1, NF))
        f          =  freq[ifreq]        
        Ibg        =  1.0e-13*Planck(f, 10000.0)  # diluted T=10000K black body as background
        if (0):  # Photons per photon package
            phot_bg    =  (Ibg*pi*2*(NX*NY+NX*NZ+NY*NZ)*GL*GL) / (h*f) / NPP
        else:     # Use *normalised* photon numbers: true photons / GL^3
            phot_bg    =  (Ibg*pi*2*(NX*NY+NX*NZ+NY*NZ)) / (h*f*GL) / NPP
        ABS[:,:,:] =  0.0        
        for pp in range(NPP):
            if (pp%1000==0): print("     pp %6d -- %5.2f per cent" % (pp, 100.0*pp/NPP))
            x, y, z, u, v, w  =  initialise_background_package(NX, NY, NZ)
            photons           =  phot_bg
            i, j, k           =  get_cell_indices(x, y, z, NX, NY, NZ)
            tau0              =  -log(rand())  # free path for scattering
            while(i>=0):   # while PP remains inside the model volume
                s             =  get_step_length(x, y, z, u, v, w)  # step
                dtaus         =  s*Ksca[ifreq]*n[i,j,k]
                if (dtaus>tau0): # --- partial step and scattering ---
                    # distance to in tau to location of scattering
                    s           =  tau0/(Ksca[ifreq]*n[i,j,k])  # distance to point of scattering
                    tau         =  s*Kabs[ifreq]*n[i,j,k]       # tau(abs.) for this step
                    ABS[i,j,k] +=  photons*(1.0-exp(-tau))
                    photons    *=  exp(-tau)                    # absorptions on the shorter step
                    u, v, w     =  isotropic_scattering(u, v, w) # all photons to new a direction
                    tau0        =  -log(rand())                 # new free path
                else:            # --- normal full step ---
                    tau         =  s*Kabs[ifreq]*n[i,j,k]       # tau =  s * Kabs * n
                    ABS[i,j,k] +=  photons*(1.0-exp(-tau))      # absorbed part
                    photons    *=  exp(-tau)                    # remaining part
                    tau0       -=  dtaus                        # tau(sca) for the full step
                x         +=  s*u                               # coordinate update
                y         +=  s*v
                z         +=  s*w
                i, j, k    =  get_cell_indices(x, y, z, NX, NY, NZ) # index update
        ABS    *=  h*freq[ifreq]   # from photon number to absorbed energy per Hz
        SUMABS +=  weights[ifreq] * ABS   # integral of absorbed energy
    print("Run time %.2f seconds" % (time.time()-t0))

    # Absorptions were the true energy per cell, divided by GL^3  ==  per unit volume
    # Divide also by H number density, to save the absorbed energy per Hydrogen atom
    SUMABS /= n
    SUMABS.tofile("absorbed.data")

    # Next calculate and plot temperatures.
    # absorbed.data is already the absorbed energy per Hydrogen atom
    # => calculate temperature using cross sections per Hydrogen atom

    # Calculate lookup table:  temperature => emitted energy  (per H)
    # Emitted energy =  Kabs0*B(T)... we can again use trapezoid integral,    
    weights =  get_integration_weights(freq)
    bins    =  300
    T       =  logspace(log10(1.0), log10(1600.0), bins)
    Eout    =  zeros(bins, float32)
    Ig      =  zeros(NF, float32)
    for i in range(bins):
        Ig      =  Kabs0*Planck(freq, T[i])    # integrand
        Eout[i] =  sum(weights * Ig)
    # Make interpolation object for conversion  energy -> temperature
    ip      =  interp1d(Eout, T, bounds_error=False, fill_value=(0.0, 1600.0))

    Ein     =  fromfile("absorbed.data", float32)
    Tdust   =  ip(Ein).reshape(NX, NY, NZ)

    print("Ein %10.3e,  Eout %10.3e ... %10.3e" % (mean(Ein), min(Eout), max(Eout)))
    
    imshow(Tdust[16,:,:])
    colorbar(label=r'$T_{\rm dust} \/ \/ \rm [K]$')
    show(block=True)
    sys.exit()
    
    


# simulate_single_frequency()
# simulate_single_frequency_2()
simulate_all_frequencies()

    
