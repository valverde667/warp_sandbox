# Example script for using the timedependentvoltage features of warp

import numpy as np
import matplotlib.pyplot as plt
import pdb

import warp as wp
from warp.utils.timedependentvoltage import TimeVoltage

# Set some useful constants
mm = 1e-3
kV = 1e3
MHz = 1e6

# ------------------------------------------------------------------------------
#    Function Defintions
# ------------------------------------------------------------------------------
def getindex(mesh, value, spacing):
    """Find index in mesh for or mesh-value closest to specified value

    Function finds index corresponding closest to 'value' in 'mesh'. The spacing
    parameter should be enough for the range [value-spacing, value+spacing] to
    encompass nearby mesh-entries .

    Parameters
    ----------
    mesh : ndarray
        1D array that will be used to find entry closest to value
    value : float
        This is the number that is searched for in mesh.
    spacing : float
        Dictates the range of values that will fall into the region holding the
        desired value in mesh. Best to overshoot with this parameter and make
        a broad range.

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        return np.where(mesh == value)[0][0]

    # Create array of possible indices
    indices = np.where((mesh > (value - spacing)) & (mesh < (value + spacing)))[0]

    # Compute differences of the indexed mesh-value with desired value
    difference = []
    for index in indices:
        diff = np.sqrt((mesh[index] ** 2 - value ** 2) ** 2)
        difference.append(diff)

    # Smallest element will be the index closest to value in indices
    i = np.argmin(difference)
    index = indices[i]

    return index


# Create function for voltage time
def voltfunc(time, Vmax=7 * kV, frequency=1 * MHz):
    return Vmax * np.cos(2 * np.pi * frequency * time)


def negvoltfunc(time, Vmax=7 * kV, frequency=1 * MHz):
    return -Vmax * np.cos(2 * np.pi * frequency * time)


# ------------------------------------------------------------------------------
#     Mesh set up
# ------------------------------------------------------------------------------
wp.w3d.xmmin = -3 * mm
wp.w3d.xmmax = 3 * mm
wp.w3d.nx = 100

wp.w3d.ymmin = -3 * mm
wp.w3d.ymmax = 3 * mm
wp.w3d.ny = 100

wp.w3d.zmmin = -2 * mm
wp.w3d.zmmax = 2 * mm
wp.w3d.nz = 200

# Set boundary conditions
wp.w3d.boundxy = wp.dirichlet
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet

# Set some simulation parameters
set_freq = 1 * MHz
set_period = 1 / set_freq
set_voltage = 7 * kV

# Set time step
nsteps = 10
wp.top.dt = set_period / nsteps

# Register the sovler being used.
solver = wp.MRBlock3D()
wp.registersolver(solver)

# ------------------------------------------------------------------------------
#    Conductor Creation
# Here two annuli are created with the let annulus being grounded and the right
# annulus having an RF applied voltage. The geometries for the conductors are
# defined here.
# ------------------------------------------------------------------------------
zextent = 0.75 * mm
rmax = 2 * mm
rmin = 0.5 * mm
separation = 1 * mm
zc = separation / 2 + zextent / 2

# Create a time varying acceleration gap with 0.5*mm thickness centered at z=0
leftplate = wp.Annulus(rmin=rmin, rmax=rmax, length=zextent, zcent=-zc, voltage=0)
rightplate = wp.Annulus(rmin=rmin, rmax=rmax, length=zextent, zcent=zc)

# Invoke the time variation. This class can be found in the warp directory in
# warp/scripts/utils. There are other settings that can be applied but this simply
# makes it so that the conductor voltage is recalculated with wp.top.time using
# the function provided.
TimeVoltage(rightplate, voltfunc=voltfunc)

wp.installconductors(leftplate)
wp.installconductors(rightplate)
wp.generate()

# ------------------------------------------------------------------------------
#    Visualization
# The field is oscillated for nsteps and the potential at each step is collected.
# During the while-loop the absolute max voltage is printed. After, the collected
# potentials are plotted toegether to show the variation.  
# ------------------------------------------------------------------------------
x = wp.w3d.xmesh
y = wp.w3d.ymesh
z = wp.w3d.zmesh

# Get index for x = 0, y = 0
xc_ind = getindex(x, 0.0, wp.w3d.dx)
yc_ind = xc_ind
wp.setup()
wp.pfzx(fill=1, filled=1)
wp.fma()

phi0 = np.zeros(shape=(nsteps, len(z)))
phi0[0, :] = wp.getphi()[xc_ind, yc_ind, :]
fig, ax = plt.subplots()
ax.axvline(x=(-zc - zextent / 2) / mm, c="k", ls="--")
ax.axvline(x=(-zc + zextent / 2) / mm, c="k", ls="--")
ax.axvline(x=(zc - zextent / 2) / mm, c="k", ls="--")
ax.axvline(x=(zc + zextent / 2) / mm, c="k", ls="--")
ax.plot(z / mm, phi0[0, :] / kV)
plt.show()

while wp.top.it < nsteps - 1:
    wp.step()
    this_phi0 = wp.getphi()[xc_ind, yc_ind, :]

    phi0[wp.top.it, :] = this_phi0
    print(max(abs(this_phi0)) / kV)

fig, ax = plt.subplots()
ax.axvline(x=(-zc - zextent / 2) / mm, c="k", ls="--")
ax.axvline(x=(-zc + zextent / 2) / mm, c="k", ls="--")
ax.axvline(x=(zc - zextent / 2) / mm, c="k", ls="--")
ax.axvline(x=(zc + zextent / 2) / mm, c="k", ls="--")
for row in phi0:
    ax.plot(z / mm, row / kV)
plt.show()
