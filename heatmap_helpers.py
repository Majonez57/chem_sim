import numpy as np


# Creates a voxel grid of reachable positions
def make_voxel_grid(points, bounds, resolution=0.05):
    #bounds should be max/min for x,y,z

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    nx,ny,nz = [int((max-min)/resolution) for (min,max) in bounds]

    grid = np.zeros((nx,ny,nz)) # Initialises our empty voxels

    # Calculate density in each voxel
    for p in points:
        x,y,z = p

        if not (xmin <= x < xmax and ymin <= y < ymax and zmin <= z < zmax):
            continue

        ix = int((x-xmin)/resolution)
        iy = int((y-ymin)/resolution)
        iz = int((z-zmin)/resolution)

        
        try:
            grid[ix,iy,iz] += 1
        except:
            continue

    return grid