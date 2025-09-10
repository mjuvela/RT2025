import sys
sys.path.append('./')
from Aux import *


NX, NY, NZ  = 32, 32, 32  #   0.0<x<NX
n      =  create_cloud(NX, NY, NZ)
kabs   =  1.0    #  optical depth for absorptions per unit step, unit density
ksca   =  1.0    # ..... A = Qsca / (Qabs+Qsca)
NP     =  20000
GL     =  1e17   # size of a single cell
# initialise PP
x, y, z,  u, v, w,  photons  = 0.0, 0.0, 0.0,   0.0, 0.0, 0.0,  0.0


# Read dust properties, Kabs = cm2/H
freq, g, Kabs, Ksca = read_dust_file("COM.dust")
# Make Kabs and Ksca optical depths for n=1, s=GL
Kabs  *= GL
Ksca  *= GL
    
# Select one frequency
ifreq = 47
# ifreq = 10
f     = freq[ifreq]

# Calculate number of photons entering the model in one second, divide by NP
#   => number of photons per single photon package
Ibg     =  1.0e-13*Planck(f, 10000.0)  # diluted T=10000K black body as background
phot_bg =  (Ibg*pi*2*(NX*NY+NX*NZ+NY*NZ)*GL*GL) / (h*f) / NP


if (0):
    # Test initialise_background_package() routine
    POS = zeros((1000, 3), float32)
    DIR = zeros((1000, 3), float32)
    for i in range(1000):
        x, y, z, u, v, w   =  initialise_background_package(NX, NY, NZ)
        photons            =  phot_bg
        POS[i,:] = [x,y,z]
        DIR[i,:] = [u,v,w]

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
    



if (0):
    # Simulation of a single frequency, store and plot the absorbed energy
    ABS = zeros((NX, NY, NZ), float32)  # array for absorbed energy
    
    for ipacket in range(NP):
        if (ipacket%1000==0):
            print("packet  %6d  --- %5.2f per cent" % (ipacket, 100.0*ipacket/NP))
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

    


if (1):
    # Simulation of a single frequency, store and plot the absorbed energy
    # Include isotropic scattering
    ABS = zeros((NX, NY, NZ), float32)  # array for absorbed energy
    
    for ipacket in range(NP):
        if (ipacket%1000==0):
            print("packet  %6d  --- %5.2f per cent" % (ipacket, 100.0*ipacket/NP))
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

    
    
