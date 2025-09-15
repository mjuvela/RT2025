import sys
sys.path.append('./')
from Aux import *
from scipy.interpolate import interp1d

NX, NY, NZ  =  32, 32, 32     # model dimensions, for coordinates 0.0<x<NX
GL          =  0.1*PC         # size of a single cell (0.1
NPP         =  10000          # number of photon packages (PP)
n           =  create_cloud(NX, NY, NZ)

# initialise PP
x, y, z,  u, v, w,  photons  = 0.0, 0.0, 0.0,   0.0, 0.0, 0.0,  0.0

# Read dust properties, Kabs in units of [m2/H]
freq, g, Kabs0, Ksca0 = read_dust_file("COM.dust")
NF = len(freq)   # number of frequencies

# Make Kabs and Ksca optical depths for unit density n=1 and distance s=GL
Kabs  = Kabs0 * GL
Ksca  = Ksca0 * GL
    
# Select one frequency... the one closest to V band
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



if (1):
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
    sys.exit()
    


if (0):
    # Simulation of a single frequency, store and plot the absorbed energy
    # Include isotropic scattering
    if (0): n *= 100.0
    ABS = zeros((NX, NY, NZ), float32)  # array for absorbed energy
    t0  = time.time()
    for pp in range(NPP):
        if (pp%1000==0): print("pp  %6d --- %5.2f per cent" % (pp, 100.0*pp/NPP))
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
    print("Run time %.2f seconds" % (time.time()-t0))
    # convert absorbed photons to absorbed energy per cell (and per 1 Hz)
    # normalised by density
    ABS *=  h*f / n

    print("--------------------------------------------------------------------------------")
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
        imshow(ABS[ix,:,:], cmap=cm.gist_stern)
        text(0.08, 0.08, 'x=%d' % ix, transform=ax.transAxes, size=16, backgroundcolor='w')
        colorbar()
    show(block=True)
    sys.exit()
    
    

    
if (0):
    # Simulation several frequencies, calculate integrated energy and save to file
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
    sys.exit()

    

if (0):
    # Calculate and plot dust temperature;
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
    
    
