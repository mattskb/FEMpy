import numpy as np
import math


def plot_mesh(vertex_to_coords, elt_to_vertex, dof_to_coords, title, figsize, savefig):
    """Plots trianglular mesh with or without DOFs indicated
    Input:
        vertex_to_coords 	- vertex number to coordinates 
        elt_to_vertex 	 	- element number to vertex number
        dof_to_coords 		- DOF number to coordinate (optional)
        figsize	    		- size of figure (optional)
        savefig 	    	- file name of saved figure (optional)
    """
    import matplotlib.pyplot as plt

    # initialize figure
    plt.figure(figsize=figsize)

    # plot mesh
    x = vertex_to_coords[:,0]; y = vertex_to_coords[:,1]
    plt.gca().set_aspect("equal")
    plt.triplot(x, y, elt_to_vertex, "g-", linewidth=.5)

    # plot DOFs
    if dof_to_coords is not None:
        dof_x = dof_to_coords[:,0]; dof_y = dof_to_coords[:,1]
        plt.plot(dof_x,dof_y,"go",markersize=20)

    # set title
    if title is None:
        if dof_to_coords is None:
            plt.title("Triangular mesh.")
        else:
            plt.title("Triangular mesh with DOFs.")
    else:
        plt.title(title)

    plt.xlabel("x")
    plt.ylabel("y")


    # save figure
    if savefig is not None:
        plt.savefig(savefig)

    plt.show()

#--------------------------------------------------------------------------------------#

def regular_mesh_data(box, res, diag):
    """Data for regular triangular mesh
    Input:
        box 				- bounding box ([start_x,end_x],[start_y,end_y])
        res 				- number of divisions of x and y (nx,ny)
        diag 				- diagonal to the left or right 
    Output:
        vertex_to_coords 	- vertex number to coordinates
        elt_to_vertex 		- element number to vertex numbers 
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
    nx_vertex = nx+1; ny_vertex = ny+1    		# number of vertex in x and y directions
    n_vertex = nx_vertex*ny_vertex              # total number of vertex
    n_elts = 2*nx*ny                            # total number of elements

    # assemble vertex_to_coords
    # numbered from bottom left, then right, then one up start from left again 
    xx = np.linspace(x[0],x[1],nx_vertex)
    yy = np.linspace(y[0],y[1],ny_vertex)
    vertex_to_coords = []

    # loop over vertexes
    for iy in range(ny_vertex):
        for ix in range(nx_vertex):
            vertex_to_coords.append([xx[ix], yy[iy]])
    vertex_to_coords = np.array(vertex_to_coords, dtype=float)
    assert len(vertex_to_coords) == n_vertex, 'Assembly of vertex_to_coords failed.'

    # assemble elt_to_vertex
    # numbering follows cell numbering (bottom left to right, one up, left to right etc.)
    # lower elt in cell comes before upper elt
    # anti-clockwise numbering of vertexes (start from bottom left)
    elt_to_vertex = []

    # loop over cells
    for iy in range(ny):
        for ix in range(nx):
            # vertexes of cell (ix,iy), from bottom left, anti-clockwise
            v0 = iy*nx_vertex+ix; v1 = v0+1
            v2 = v0+nx_vertex; v3 = v2+1

            if diag in {'right', 'r'}:
                elt_to_vertex.append([v0, v1, v3]) 
                elt_to_vertex.append([v0, v3, v2])
            else:
                elt_to_vertex.append([v0, v1, v2])
                elt_to_vertex.append([v1, v3, v2])
    
    elt_to_vertex = np.array(elt_to_vertex, dtype=int)
    assert len(elt_to_vertex) == n_elts, 'Assembly of elt_to_vertex failed.'

    return vertex_to_coords, elt_to_vertex

#--------------------------------------------------------------------------------------#

def regular_mesh_dofs(box, res, deg, diag):
    """Lagrange DOF data for regular triangular mesh
    Input:
        box                 - bounding box ([start_x,end_x],[start_y,end_y])
        res                 - number of divisions of x and y (nx,ny)
        deg                 - degree of Lagrange DOFs
        diag                - diagonal to the left or right 
    Output:
        def_to_coords       - vertex number to coordinates
        elt_to_dofs         - element number to vertex numbers 
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

    # assemble dof_to_coords
    # numbered from bottom left, then right, then one up start from left again 
    xx = np.linspace(x[0],x[1],nx_dofs)
    yy = np.linspace(y[0],y[1],ny_dofs)
    dof_to_coords = []

    # loop over dofs
    for iy in range(ny_dofs):
        for ix in range(nx_dofs):
            dof_to_coords.append([xx[ix], yy[iy]])
    assert len(dof_to_coords) == n_dofs, 'Assembly of vertex_to_coords failed.'

    # assemble elt_to_dofs
    # numbering follows cell numbering (bottom left to right, one up, left to right etc.)
    # lower elt in cell comes before upper elt
    # anti-clockwise numbering of vertexes (start from bottom left)
    elt_to_dofs = []

    # loop over cells
    for iy in range(ny):
        for ix in range(nx):
            # dofs in current cell as matrix
            dofs_in_cell = np.array([[nx_dofs*(deg*iy + i) + deg*ix + j \
                                        for j in range(deg+1)] for i in range(deg,-1,-1)])

            # extract dofs in the upper and lower element
            if diag in {"right","r"}: 
                dofs_in_cell = np.fliplr(dofs_in_cell)

            elt_to_dofs.append(dofs_in_cell[np.tril_indices(deg+1)])
            elt_to_dofs.append(dofs_in_cell[np.triu_indices(deg+1)])
    assert len(elt_to_dofs) == n_elts, 'Assembly of elt_to_vertex failed.'

    return np.array(dof_to_coords, dtype=float), np.array(elt_to_dofs, dtype=int)

#--------------------------------------------------------------------------------------#

class Mesh:
    """Triangular mesh """
    def __init__(self, vertex_to_coords, elt_to_vertex):
        """
        Input:
            vertex_to_coords        - vertex number to coordinates, i.e., np.array([[x, y]], float) 
            elt_to_vertex           - element number to vertex numbers, i.e., np.array([[v0, v1, v2]], int) 
        """

        # set data
        self.__vertex_to_coords = vertex_to_coords
        self.__elt_to_vertex = elt_to_vertex

    def n_elts(self):
        """ return number of elements in mesh """
        return self.__elt_to_vertex.shape[0]

    def n_vertices(self):
        """ return number of vertices in mesh """
        return self.__vertex_to_coords.shape[0]

    def n_edges(self):
        """ return number of edges (Eulers formula) """
        return self.n_elts() + self.n_vertices() - 1

    def vertex_to_coords(self, n=None):
        """Get vertex number to coordinates (global or of vertex n) """
        if n is None:
            return self.__vertex_to_coords
        else:
            return self.__vertex_to_coords[n,:]

    def elt_to_vertex(self, n=None):
        """ get elt number to vertex numbers (global or of elt n) """
        if n is None:
            return self.__elt_to_vertex
        else:
            return self.__elt_to_vertex[n,:]

    def elt_to_vcoords(self, n=None):
        """return element number to vertex coords, or vertex coords of element n 
        i.,e, n_eltsx3x2 or 3x2 array """
        if n is None:
            return self.__vertex_to_coords[self.__elt_to_vertex]
        else:
            return self.__vertex_to_coords[self.__elt_to_vertex[n,:]]

    def elt_to_ccoords(self, n=None):
        """ return element number to center coords or center coord of element n """
        if n is None:
            return np.sum(self.elt_to_vcoords(),axis=1)/3  
        else:
            return np.sum(self.elt_to_vcoords(n),axis=0)/3
            
    def plot(self, dof_to_coords=None, title=None, figsize=(10,10), savefig=None):
        """ plot mesh
        DOFs given by dofs can be shown in figure
        file - name of .pdf file if figure is saved
        """
        plot_mesh(self.__vertex_to_coords, self.__elt_to_vertex, dof_to_coords,
            title, figsize, savefig)

#--------------------------------------------------------------------------------------#

class RectangleMesh(Mesh):
    """Rectangle shaped triangular mesh (unit square is default) """
    def __init__(self, x=[0,1], y=[0,1], nx=4, ny=4, diag='right'):
        """
        Input: 
            x,y             - intervals defining rectangle box
            nx,ny           - subdivisions in x and y directions
            diag            - left or right 
        """
        # get data
        vertex_to_coords, elt_to_vertex = regular_mesh_data((x, y), (nx, ny), diag)

        super(RectangleMesh, self).__init__(vertex_to_coords, elt_to_vertex)
        self.__box = (x, y)
        self.__res = (nx, ny)
        self.__diag = diag

    def mesh_size(self):
        """ return h - size of elements in mesh """
        return max((self.__box[0][1] - self.__box[0][0])/float(self.__res[0]),\
                   (self.__box[1][1] - self.__box[1][0])/float(self.__res[1]))

#--------------------------------------------------------------------------------------#

class TriangleMesh(Mesh):
    """ triangle mesh for plotting shape functions """
    def __init__(self, nx=4, ny=4):
        
        nx_vertex = nx+1; ny_vertex = ny+1          # number of vertex in x and y directions
        n_vertex = nx_vertex*ny_vertex              # total number of vertex
        n_elts = 2*nx*ny                            # total number of elements

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
        #assert len(elt_to_vertex) == n_elts, 'Assembly of elt_to_vertex failed.'


        super(TriangleMesh, self).__init__(np.array(vertex_to_coords), np.array(elt_to_vertex))

#-------------------------------------------------------------------------------------#

if __name__ == '__main__':

    # mesh
    nx = 1; ny = 1; diag = "l"; deg = 10
    mesh = RectangleMesh(nx=nx,ny=ny,diag=diag)
    dof_to_coords, elt_to_dofs = regular_mesh_dofs(([1,0],[1,0]), (nx,ny), deg, diag)
    print(elt_to_dofs)
    mesh.plot(dof_to_coords=dof_to_coords)
    
    
    #print(mesh.vertex_to_coords())
    #print(mesh.elt_to_vertex())
    #print(mesh.elt_to_vcoords(0))
    #print(mesh.elt_to_ccoords())


    #mesh.plot()

    mesh = TriangleMesh(nx=10,ny=10)


    #print(mesh.vertex_to_coords().shape)
    #print(mesh.elt_to_vertex())
    #mesh.plot()

    #dof_to_coords, elt_to_dofs = regular_mesh_dofs(([1,0],[1,0]), (2,2), 2, "right")