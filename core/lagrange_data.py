"""
Generate lagrange data for regular triangular mesh
"""
import numpy as np

def regular_mesh_data(box, res, deg, diag):
    """Lagrange DOF data for regular triangular mesh
    Input:
        box                  - bounding box ([start_x,end_x],[start_y,end_y])
        res                  - number of divisions of x and y intervals (nx,ny)
        deg                  - degree of Lagrange DOFs
        diag                 - diagonal to the left or right 
    Output:
        node_to_coords       - node number to coordinates (for deg=1 nodes are element vertices)
        elt_to_nodes         - element number to node numbers 
    """
    # check valid data
    assert len(box) == len(res), 'Incompatible box and res arguments.'
    if len(box)!= 2:
        raise ValueError('Invalid box and res arguments.')

    # check valid diagonal arg
    diag = diag.lower()
    assert diag in {'left','l','right','r'}, 'Invalid diagonal argument.'
    
    x = box[0]; y = box[1]                      # bounding box
    nx = res[0]; ny = res[1]                    # number of cells in each direction (2 elts per cell)
    nx_dofs = deg*nx+1; ny_dofs = deg*ny+1      # number of vertex in x and y directions
    n_dofs = nx_dofs*ny_dofs                    # total number of vertex
    n_elts = 2*nx*ny                            # total number of elements
    n_boundary_edges = 2*nx + 2*ny              # total number of boundary edges

	# assemble node_to_coords
	# numbered from bottom left, then right, then one up start from left again 
    xx = np.linspace(x[0],x[1],nx_dofs)
    yy = np.linspace(y[0],y[1],ny_dofs)
    node_to_coords = []

	# loop over dofs
    for iy in range(ny_dofs):
        for ix in range(nx_dofs):
            node_to_coords.append([xx[ix], yy[iy]])
    assert len(node_to_coords) == n_dofs, 'Assembly of vertex_to_coords failed.'

    # assemble elt_to_nodes
    # numbering follows cell numbering (bottom left to right, one up, left to right etc.)
    # lower elt in cell comes before upper elt
    # anti-clockwise numbering of vertexes (start from bottom left)
    elt_to_nodes = []

    # boundary edge to node numbers (divided in four groups corresponding to sides of rectangle)
    bottom = []; top = []; right = []; left = []

    # loop over cells
    for iy in range(ny):
        for ix in range(nx):
            # dofs in current cell as matrix
            dofs_in_cell = np.array([[nx_dofs*(deg*iy + i) + deg*ix + j \
                                        for j in range(deg+1)] for i in range(deg,-1,-1)])

            # extract dofs in the upper and lower element
            if diag in {"right","r"}: 
                # når diag=r så er øverste element riktig fortegn
                dofs_in_cell = np.fliplr(dofs_in_cell)

                # lower element
                elt_to_nodes.append(dofs_in_cell[np.tril_indices(deg+1)]) 

                # upper element
                elt_to_nodes.append(np.flip(dofs_in_cell[np.triu_indices(deg+1)]))
            else:
                # når diag=l så er nederste element riktig fortegn

                # lower element
                elt_to_nodes.append(dofs_in_cell[np.tril_indices(deg+1)])
                # upper element
                elt_to_nodes.append(np.flip(dofs_in_cell[np.triu_indices(deg+1)])) 

            #elt_to_nodes.append(dofs_in_cell[np.tril_indices(deg+1)])
            #elt_to_nodes.append(dofs_in_cell[np.triu_indices(deg+1)])

            if iy == 0: bottom.append(dofs_in_cell[-1,:])
            if ix == 0: left.append(dofs_in_cell[:,0])
            if iy == ny-1: top.append(dofs_in_cell[0,:])
            if ix == nx-1: right.append(dofs_in_cell[:,-1])

    assert len(elt_to_nodes) == n_elts, "Assembly of elt_to_vertex failed."
    assert len(bottom+left+top+right) == n_boundary_edges, "Assembly of boundary failed."
    boundary = {"bottom": np.array(bottom),
                "left":   np.array(left),
                "top":    np.array(top),
                "right":  np.array(right)}

    return np.array(node_to_coords, dtype=float), np.array(elt_to_nodes, dtype=int), boundary


def triangle_mesh_data(nx, ny):
	"""Lagrange DOF data for regular mesh of reference triangle, i.e., 
	triangle with vertices {(0,0), (0,1), (1,0)}
    Input:
        res                   - number of divisions of x and y intervals (nx,ny)
    Output:
        vertex_to_coords      - node number to coordinates (for deg=1 nodes are element vertices)
        elt_to_vertex         - element number to node numbers 
    """
	nx_vertex = nx+1; ny_vertex = ny+1          # number of vertex in x and y directions
	n_vertex = nx_vertex*ny_vertex              # total number of vertex
	n_elts = nx*ny                              # total number of elements

	# assemble vertex_to_coords
	# numbered from bottom left, then right, then one up start from left again 
	xx = np.linspace(0,1,nx_vertex)
	yy = np.linspace(0,1,ny_vertex)
	vertex_to_coords = []

	# loop over vertexes
	for iy in range(ny_vertex):
	    for ix in range(nx_vertex-iy):
	        vertex_to_coords.append([xx[ix], yy[iy]])
	vertex_to_coords = np.array(vertex_to_coords, dtype=float)
	#assert len(vertex_to_coords) == n_vertex, 'Assembly of vertex_to_coords failed.'

	# assemble elt_to_vertex
	# numbering follows cell numbering (bottom left to right, one up, left to right etc.)
	# lower elt in cell comes before upper elt
	# anti-clockwise numbering of vertexes (start from bottom left)
	elt_to_vertex = []

	# loop over cells
	for iy in range(ny):
	    for ix in range(nx-iy):
	        # vertexes of cell (ix,iy), from bottom left, anti-clockwise
	        v0 = sum([nx_vertex - iiy for iiy in range(iy)]) + ix 
	        v1 = v0+1
	        v2 = v0+nx_vertex-iy
	        v3 = v2+1

	        elt_to_vertex.append([v0, v1, v2])
	        if ix < nx - iy - 1:
	            elt_to_vertex.append([v1, v3, v2])

	elt_to_vertex = np.array(elt_to_vertex, dtype=int)
	assert len(elt_to_vertex) == n_elts, 'Assembly of elt_to_vertex failed.'

	return vertex_to_coords, elt_to_vertex