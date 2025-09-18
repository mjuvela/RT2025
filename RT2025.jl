using_pyplot = true  # plotting with Plots or with PyPlot

using Plots       
using PyPlot
using Measures
using Printf
using Interpolations

const c0    =  2.99792e+8
const h0    =  6.62606957e-34
const k0    =  1.3806488e-23
const EPS   =  1.0e-5
const PC    =  3.0856775e+15


function Planck(f, T)
    f64 = Float64.(f)
    return  (2.0*h0*(f64.^3/c0^2))  ./  (  exp.((h0*f64)./(k0*T)) - 1.0 )
end


function create_cloud(NX, NY, NZ)
    # Generate density distribution for the model
    # Peak density 1e4 cm-3  =  1e10 m-3
    n = zeros(Float32, NX, NY, NZ)
    for i=1:NX
        dx = i-0.5*(NX+1)
        for j=1:NY
            dy = j-0.5*(NY+1)
            for k=1:NZ
                dz = k-0.5*(NZ+1)
                n[i,j,k] = 1.0e10 * exp(-0.005*(dx^2+dy^2+dz^2))
            end
        end
    end
    return n
end


function initialise_background_package(NX, NY, NZ)
    # Direction (u,v,w), position (x, y, z)
    u    =   rand()
    ct   =   sqrt(rand())
    st   =   sqrt(1.0-ct^2)
    phi  =   2.0*pi*rand()
    cp   =   cos(phi)
    sp   =   sin(phi)
    if (u<1/6)     #  side X=0
        x, y, z =  EPS,       NY*rand(), NZ*rand()
        u, v, w =  ct,        st*cp,     st*sp
    elseif (u<2/6) # side X=NX
        x, y, z =  NX-EPS,    NY*rand(), NZ*rand()
        u, v, w = -ct,        st*cp,     st*sp
    elseif (u<3/6) # side Y=0
        x, y, z =  NX*rand(), EPS,       NZ*rand()
        u, v, w =  st*cp,     ct,        st*sp
    elseif (u<4/6) # side Y=NY
        x, y, z =  NX*rand(), NY-EPS,    NZ*rand()
        u, v, w =  st*cp,     -ct,       st*sp
    elseif (u<5/6) # side Z=0
        x, y, z =  NX*rand(), NY*rand(), EPS
        u, v, w =  st*cp,     st*sp,     ct
    else           # side Z=NZ
        x, y, z =  NX*rand(), NY*rand(), NX-EPS
        u, v, w =  st*cp,     st*sp,     -ct
    end
    return x, y, z,   u, v, w
end                                        



function get_step_length(x, y, z,  u, v, w)
    # return distance to next cell boundary in direction (u, v, w),
    # positiong (x,y,z) is not updated
    if (u>0.0)
        dx  =  (1.0+EPS-mod(x, 1.0)) / u
    else
        dx  =     -(EPS+mod(x, 1.0)) / u
    end
    if (v>0.0)
        dy  =  (1.0+EPS-mod(y, 1.0)) / v
    else
        dy  =     -(EPS+mod(y, 1.0)) / v
    end
    if (w>0.0)
        dz  =  (1.0+EPS-mod(z, 1.0)) / w
    else
        dz  =     -(EPS+mod(z, 1.0)) / w
    end
    return min(dx, dy, dz)
end



function get_cell_indices(x, y, z, NX, NY, NZ)
    # Return integer cell indices for position (x, y, z), e.g. first index 1 <= i <= NX
    if ((x<=0.0)|(x>=NX))  return -1, -1, -1   end
    if ((y<=0.0)|(y>=NY))  return -1, -1, -1   end
    if ((z<=0.0)|(z>=NZ))  return -1, -1, -1   end
    return  1+Int32(floor(x)), 1+Int32(floor(y)), 1+Int32(floor(z))
end



function read_dust_file(name)
    freq, g, Kabs, Ksca = [], [], [], []
    open(name, "r") do fp
        readline(fp)   # skip line
        A = parse(Float32, split(readline(fp))[1])         # dust-to-gas number ratio
        B = parse(Float32, split(readline(fp))[1]) * 0.01  # grain size [m]
        N = parse(Int32, split(readline(fp))[1])
        freq = zeros(Float32, N)
        g    = zeros(Float32, N)
        Kabs = zeros(Float32, N)
        Ksca = zeros(Float32, N)
        for i=1:N
            s        = split(readline(fp))
            freq[i]  =  parse(Float32, s[1])
            g[i]     =  parse(Float32, s[2])
            Kabs[i]  =  parse(Float32, s[3]) * A * pi*B^2
            Ksca[i]  =  parse(Float32, s[4]) * A * pi*B^2
        end
    end
    return freq, g, Kabs, Ksca
end


function isotropic_scattering(u, v, w)
    # return new random direction
    cos_theta =  -1.0+2.0*rand()
    sin_theta =  sqrt(1.0-cos_theta^2)
    phi       =  2.0*pi*rand()
    u         =  sin_theta*cos(phi)
    v         =  sin_theta*sin(phi)
    w         =  cos_theta
    return u, v, w
end


function get_integration_weights(freq)
    #=
    Return integration weights for trapezoidal integration over frequency axis,
    given the frequency grid freq.
    =#
    N       =  length(freq)
    weights =  zeros(Float32, N)
    for i=1:N            #  loop over intervals
        if (i>1)         #  integral +=  0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
            weights[i] += 0.5*(freq[i]-freq[i-1])
        end
        if (i<N)         #  integral +=  0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
            weights[i] += 0.5*(freq[i+1]-freq[i])
        end
    end
    return weights
end




# ###############################################################################
# -------------------------------------------------------------------------------
# ###############################################################################



NX, NY, NZ   =  32, 32,  32    # dimensions of the model grid
GL           =  0.1*PC         # physical size of cell
NPP          =  10000          # number of photon packages
n            =  create_cloud(NX, NY, NZ)

# Read dust properties from file
freq, g, Kabs0, Ksca0 = read_dust_file("COM.dust")
NF  =  length(freq)     # number of frequencies

# Make Kabs and Ksca optical depths for unit density n=1 and distance s=GL
Kabs  = Kabs0 * GL
Ksca  = Ksca0 * GL


# Select one frequency... the one closest to V band
ifreq =  argmin(abs.(freq.-c0/0.55e-6))
f     =  freq[ifreq]

# Calculate number of photons entering the model in one second, divide by NPP
#   => number of photons per single photon package
Ibg     =  1.0e-13 * Planck(f, 10000.0)  # diluted T=10000K black body as background
phot_bg =  (Ibg*pi*2*(NX*NY+NX*NZ+NY*NZ)*GL*GL) / (h0*f) / NPP

@printf("freq %.3e       Ibg %12.4e         phot_bg %.4e\n", f, Ibg, phot_bg)


# ---------------------------------------------------------------------
# Below follows a series of different runs, to be tested one at a time
# ---------------------------------------------------------------------



function simulate_single_frequency()
    # Simulation of a single frequency, store and plot the absorbed energy
    NPP = 40000
    ABS = zeros(Float32, NX, NY, NZ)  # array for absorbed energy
    t0  = time()
    for ipacket=1:NPP
        if (mod(ipacket,2000)==0)
            @printf("packet  %6d  --- %5.2f per cent\n", ipacket, 100.0*ipacket/NPP)
        end
        x, y, z, u, v, w  =  initialise_background_package(NX, NY, NZ)
        photons           =  phot_bg
        i, j, k           =  get_cell_indices(x, y, z, NX, NY, NZ)
        while(i>0)    # while PP remains inside the model volume
            s             =  get_step_length(x, y, z, u, v, w)  # step
            tau           =  s*Kabs[ifreq]*n[i,j,k]             # tau =  s * Kabs * n
            ABS[i,j,k]   +=  photons*(1.0-exp(-tau))            # absorbed part
            photons      *=  exp(-tau)                          # remaining part
            x            +=  s*u                                # coordinate update
            y            +=  s*v
            z            +=  s*w
            i, j, k       =  get_cell_indices(x, y, z, NX, NY, NZ) # index update
            # @printf(" %2d %2d %2d   %6.3f %10.3e  %10.3e %10.3e\n", i, j, k,   s, tau,  photons, phot_bg)
        end
    end # end for ipacket
    @printf("Run time %.2f seconds\n", time()-t0)

    # Convert absorbed photons to absorbed energy per cell, per unit density, per 1 Hz.
    ABS .*=  h0*f ./ n

    if using_pyplot==false  # using Plots
        # show cross section of the absorbed energy
        P = []
        for i=1:3
            ix = [2, 7, Int32(NX/2)][i]
            push!(P, heatmap(ABS[ix,:,:], title=@sprintf("x=%d", ix)))
        end
        l = @layout [a{0.3w, 0.9h} b{0.3w, 0.9h} c{0.3w, 0.9h}]
        pl = plot(P[1], P[2], P[3], layout=l, size=(1000, 230), margin=4mm)
        # pl = plot(P[1], P[2], P[3], layout=grid(1,3), size=(1200, 500))
        display(pl)
    else  # else using PyPlot
        clf()
        for i=1:3
            ix = [2, 7, Int32(NX/2)][i]
            subplot(1,3,i)
            imshow(ABS[ix,:,:])
            title(@sprintf("x=%d", ix))
        end
        
    end
end



function simulate_single_frequency_2()
    # Simulation of a single frequency, store and plot the absorbed energy
    # Include isotropic scattering and russian roulette
    NPP = 40000
    ABS = zeros(Float32, NX, NY, NZ)  # array for absorbed energy
    t0  = time()
    for ipacket=1:NPP
        if (mod(ipacket,2000)==0)
            @printf("packet  %6d  --- %5.2f per cent\n", ipacket, 100.0*ipacket/NPP)
        end
        x, y, z, u, v, w  =  initialise_background_package(NX, NY, NZ)
        photons           =  phot_bg
        i, j, k           =  get_cell_indices(x, y, z, NX, NY, NZ)
        tau0              = -log(rand())  #  free path for scattering
        while(i>0)    # while PP remains inside the model volume
            s             =  get_step_length(x, y, z, u, v, w)  # step
            dtaus         =  s*Kabs[ifreq]*n[i,j,k]             # tau =  s * Kabs * n
            if (dtaus>tau0) #  will scatter before the end of the fulls tep
                s           =  tau0/(Ksca[ifreq]*n[i,j,k])      # distance to scattering
                tau         =  s*Kabs[ifreq]*n[i,j,k]           # tau(absorptions) for the step
                ABS[i,j,k] +=  photons*(1.0-exp(-tau))          # absorbed part
                photons    *=  exp(-tau)                        # remaining part
                u, v, w     =  isotropic_scattering(u,v,w)      # new direction
                tau0        =  -log(rand())                     # new free paths
            else  # normal full step
                tau         =  s*Kabs[ifreq]*n[i,j,k]
                ABS[i,j,k] +=  photons*(1.0-exp(-tau))
                photons    *=  exp(-tau)
                tau0       -=  dtaus                            # remaining part of free path
            end
            x            +=  s*u                                # coordinate update
            y            +=  s*v
            z            +=  s*w
            i, j, k       =  get_cell_indices(x, y, z, NX, NY, NZ) # index update
            # @printf(" %2d %2d %2d   %6.3f %10.3e  %10.3e %10.3e\n", i, j, k,   s, tau,  photons, phot_bg)
            # The Russian roulette
            if (photons<1.0e-3*phot_bg)
                if (rand()<0.5)
                    break
                else
                    photons *= 2.0
                end
            end
        end  # end of while i>0
    end # end for ipacket
    @printf("Run time %.2f seconds\n", time()-t0)

    # Convert photon numbers to absorbed energy per cell, unit density, 1 Hz interval
    ABS .*=  (h0*f) ./ n

    # Plot cross section of the absorbed energy    
    if using_pyplot==false
        P = []
        for i=1:3
            ix = [2, 7, Int32(NX/2)][i]
            push!(P, heatmap(ABS[ix,:,:], title=@sprintf("x=%d", ix)))
        end
        l = @layout [a{0.3w, 0.9h} b{0.3w, 0.9h} c{0.3w, 0.9h}]
        pl = plot(P[1], P[2], P[3], layout=l, size=(1000, 230), margin=4mm)
        # pl = plot(P[1], P[2], P[3], layout=grid(1,3), size=(1200, 500))
        display(pl)
    else
        clf()
        for i=1:3
            ix = [2, 7, Int32(NX/2)][i]
            subplot(1,3,i)
            imshow(ABS[ix,:,:])
            title(@sprintf("x=%d", ix))
        end
    end
end




function simulate_all_frequencies()
    # Simulation with isotropic scattering and russian roulette,
    # loop over all frequencies and save integral of absorbed energy.
    SUMABS  =  zeros(Float32, NX, NY, NZ)  # total absorbed energy
    ABS     =  zeros(Float32, NX, NY, NZ)  # absorbed energy per frequency, per 1Hz
    weights =  get_integration_weights(freq)
    NPP     =  2000    
    t0      =  time()
    for ifreq=1:NF
        @printf("Frequency %3d / %3d\n", ifreq, NF)
        local f = freq[ifreq]
        local Ibg = 1.0e-13*Planck(f, 10000.0)   # background diluted T=10000K blackbody
        # Use rescaled photon numbers = original / GL^3  => result will be per unit volume !!
        local phot_bg   =  (Ibg*pi*2*(NX*NY+NX*NZ+NY*NZ)) / (h0*f*GL) / NPP
        ABS[:,:,:] .= 0.0
        for pp=1:NPP   # loop over simulated photon packages        
            x, y, z, u, v, w  =  initialise_background_package(NX, NY, NZ)
            photons           =  phot_bg
            i, j, k           =  get_cell_indices(x, y, z, NX, NY, NZ)
            tau0              = -log(rand())  #  free path for scattering
            while(i>0)    # while PP remains inside the model volume
                s             =  get_step_length(x, y, z, u, v, w)  # step
                dtaus         =  s*Kabs[ifreq]*n[i,j,k]             # tau =  s * Kabs * n
                if (dtaus>tau0) #  will scatter before the end of the fulls tep
                    s           =  tau0/(Ksca[ifreq]*n[i,j,k])      # distance to scattering
                    tau         =  s*Kabs[ifreq]*n[i,j,k]           # tau(absorptions) for the step
                    ABS[i,j,k] +=  photons*(1.0-exp(-tau))          # absorbed part
                    photons    *=  exp(-tau)                        # remaining part
                    u, v, w     =  isotropic_scattering(u,v,w)      # new direction
                    tau0        =  -log(rand())                     # new free paths
                else  # normal full step
                    tau         =  s*Kabs[ifreq]*n[i,j,k]
                    ABS[i,j,k] +=  photons*(1.0-exp(-tau))
                    photons    *=  exp(-tau)
                    tau0       -=  dtaus                            # remaining part of free path
                end
                x            +=  s*u                                # coordinate update
                y            +=  s*v
                z            +=  s*w
                i, j, k       =  get_cell_indices(x, y, z, NX, NY, NZ) # index update
                # @printf(" %2d %2d %2d   %6.3f %10.3e  %10.3e %10.3e\n", i, j, k,   s, tau,  photons, phot_bg)
                # The Russian roulette
                if (photons<1.0e-3*phot_bg)
                    if (rand()<0.5)
                        break
                    else
                        photons *= 2.0
                    end
                end
            end  # end of while i>0
        end # end for pp loop
        # Convert photon numbers to absorbed energy per unit density and 1 Hz interval
        ABS    .*=  (h0*f) ./ n
        SUMABS .+= weights[ifreq].*ABS    # cumulative value of the integral in each cell
    end # loop over frequencies
    @printf("Run time %.2f seconds\n", time()-t0)    
    # Save absorptions to a binary file
    write("absorbed_jl.data", SUMABS)
end


function calculate_temperatures()
    # Read the above-saved absorbed_jl.data file and calculate dust temperatures.
    # File has already total absorbed energy per Hydrogen atom.

    # Start by calculating lookup table:  temperature <-- energy
    weights = get_integration_weights(freq)  # we use same trapezoid integration here
    bins    = 300
    T       = Float32.(logrange(1.0, 1600.0, bins))  # grid of temperature values
    Eout    = zeros(Float32, bins)                   # vector for emitted energies
    for i=1:bins
        Ig  =  Kabs0.*Planck.(freq, T[i])            # integrand
        Eout[i]   =  sum(weights.*Ig)                # emitted energy for each temperature
    end
    @printf("T = %.2f ... %.2f   =>  E = %.3e ... %.3e\n", T[1], T[end], Eout[1], Eout[end])
    # Prepare to interpolate  E -> T
    ip    =  Interpolations.linear_interpolation(Eout, T, extrapolation_bc=NaN)    
    Ein   =  zeros(Float32, NX, NY, NZ)
    read!("absorbed_jl.data", Ein)
    Tdust =  ip.(Ein)
    
    # Plot cross section of the temperature field
    if using_pyplot==false
        P = heatmap(Tdust[16,:,:], size=[1000,300])
        display(P)
    else
        clf()
        imshow(Tdust[16,:,:])
    end
end


# *** select one of the following ***
CASE = 3


if (CASE==1)
    simulate_single_frequency()
elseif (CASE==2)
    simulate_single_frequency_2()
elseif (CASE==3)
    simulate_all_frequencies()
    calculate_temperatures()
end
