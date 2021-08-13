import numpy as np
import math


def plot_mesh(vertex_to_coords, elt_to_vertex, dof_to_coords=None, figsize=(8,6), savefig=None):
    """Plots trianglular mesh defined with or without dof overlay
    Input:
        vertex_to_coords 	- vertex number to coordinates 
        elt_to_vertex 	 	- element number to vertex number
        dof_to_coords 		- DOF number to coordinate (optional)
        figsize				- size of figure (optional)
        savefig 			- file name of saved figure (optional)
    """
    import matplotlib.pyplot as plt

    if dof_to_coords:
        plt.figure('Mesh plot with DOFs', figsize=figsize)
    else:
        plt.figure('Mesh plot', figsize=figsize)


    # plot mesh
    x = vertex_to_coords[:,0]; y = vertex_to_coords[:,1]
    plt.gca().set_aspect('equal')
    plt.triplot(x, y, elt_to_vertex[:,:3], 'g-', linewidth=.5)

    # plot DOFs
    if dof_to_coords:
        dof_x = dof_to_coords[:,0]; dof_y = dof_to_coords[:,1]
        plt.plot(dof_x,dof_y,'go')
        plt.title('Triangular mesh with DOFs.')
    else:
        plt.title('Triangular mesh.')

    plt.xlabel('x')
    plt.ylabel('y')


    # save figure
    if savefig:
        plt.savefig(savefig)

    plt.show()

def regular_mesh_data(box=([0,1],[0,1]), res=(4,4), deg=1, diag='right'):
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
    if len(box != 2)
    	raise ValueError('Invalid box and res arguments.')

    # check valid diagonal arg
    diag = diag.lower()
    assert diag in {'left','l','right','r'}, 'Invalid diagonal argument.'
    
    x = box[0]; y = box[1]                      # bounding box
    nx = res[0]; ny = res[1]                    # number of divisions of box in each direction
    nx_vertex = nx+1; ny_vertex = ny+1    		# number of vertex in x and y directions
    n_vertex = nx_vertex*ny_vertex              # total number of vertex
    n_elts = 2*nx*ny                            # total number of elements

    # assemble vertex_to_coords
    xx = np.linspace(x[0],x[1],nx_vertex)
    yy = np.linspace(y[0],y[1],ny_vertex)
    vertex_to_coords = []
    for iy in range(ny_vertex):
        for ix in range(nx_vertex):
            vertex_to_coords.append([xx[ix], yy[iy]])
    vertex_to_coords = np.array(vertex_to_coords, dtype=float)
    assert len(vertex_to_coords) == n_vertex, 'Assembly of vertex_to_coords failed.'

    # assemble elt_to_vertex, anti-clockwise numbering of nodes (start from bottom left)
    elt_to_vertex = []

    for iy in range(ny):
        for ix in range(nx):
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

def regular_mesh_bdata(res=[4,4]):
    """Boundary data for regular triangular mesh 
    Input:
        res - number of divisions of x and y (nx,ny)
    Output:
        boundary - dictionary of edge number to node numbers matrices
    """


    # raise ValueError('Invalid res argument.')
    
    nx = res[0]; ny = res[1]                    # number of divisions of interval
    n_bedges = 2*nx + 2*ny                      # number of boundary edges
    nx_vertex = deg*nx+1; ny_vertex = deg*ny+1    # number of nodes in x and y directions

    bottom = []; top = []; right = []; left = []
    # assemble edge_to_nodes
    for n in range(nx):
        n *= deg
        bnodes = [n+i for i in range(deg+1)]
        bottom.append([bnodes[0], bnodes[-1]] + [bnodes[i] for i in range(1,deg)])

        tnodes = [nx_vertex*ny_vertex - (n+i) for i in range(1,deg+2)]
        #top.append([tnodes[0], tnodes[-1]] + [tnodes[i] for i in range(1,deg)])
        top.append([tnodes[-1], tnodes[0]] + [tnodes[i] for i in range(1,deg)])

    for n in range(ny):
        rnodes = [(n*deg+i)*nx_vertex-1 for i in range(1,deg+2)]
        right.append([rnodes[0], rnodes[-1]] + [rnodes[i] for i in range(1,len(rnodes)-1)])

        lnodes = [(n*deg+i)*nx_vertex for i in range(deg+1)]
        left.append([lnodes[-1], lnodes[0]] + [lnodes[i] for i in range(1,len(lnodes)-1)])

    top.reverse(); #left.reverse()
    boundary = {'bottom':bottom, 'top':top, 'right':right, 'left':left}


        

    return boundary

#--------------------------------------------------------------------------------------#

class SuperMesh(object):
    """ super class for meshes """
    def __init__(self, vertex_to_coords, elt_to_vertex, boundary=None):
        """
        Input:
            vertex_to_coords - node number to coordinates matrix
            elt_to_vertex - elt number to node numbers matrix
            boundary - dictionary of boundary edge to nodes matrices (optional)
        """

        elt_to_vcoords = vertex_to_coords[elt_to_vertex[:,:3]]         # elt number to vertex coords
        n_elts = elt_to_vertex.shape[0]                                  # number of elements
        n_vertices = vertex_to_coords.shape[0]                            # number of vertices
        n_edges = n_elts + n_vertices - 1                               # number of edges (Eulers formula for 2D)

        # set data
        self.__vertex_to_coords = vertex_to_coords
        self.__elt_to_vertex = elt_to_vertex
        self.__elt_to_vcoords = elt_to_vcoords
        self.__boundary = boundary
        self.__n_elts = n_elts
        self.__n_vertices = n_vertices
        self.__n_edges = n_edges

    def n_elts(self):
        """ return number of elements in mesh """
        return self.__n_elts

    def n_vertices(self):
        """ return number of vertices in mesh """
        return self.__n_vertices

    def n_edges(self):
        """ return number of edges (only for 2D mesh) """
        return self.__n_edges

    def vertex_to_coords(self):
        """ get node number to coordinate matrix """
        return self.__vertex_to_coords

    def elt_to_vertex(self):
        """ get elt number to node numbers matrix """
        return self.__elt_to_vertex

    def elt_to_vcoords(self, n=None):
        """ return element number to vertex coords matrix, or vertex coords of element n """
        if n is None:
            return self.__elt_to_vcoords
        else:
            return self.__elt_to_vcoords[n]

    def elt_to_ccoords(self, n=None):
        """ return element number to center coords matrix or center coord of element n """
        if n is None:
            return np.array([sum(self.elt_to_vcoords(j))/(self.dim()+1) for j in range(self.n_elts())])
        else:
            return sum(self.elt_to_vcoords(n))/(self.dim()+1)


    def plot(self, dofs=None, file=None):
        """ plot figure of mesh
        DOFs given by dofs can be shown in figure
        file - name of .pdf file if figure is saved
        """
        plot_mesh(self.__vertex_to_coords, self.__elt_to_vertex, dofs=dofs, file=file)

#--------------------------------------------------------------------------------------#

class RegularMesh(SuperMesh):
    """ super class for regular meshes """
    def __init__(self, box=[[0,1], [0,1]], res=[4,4], diag='right'):
        """
        Input: (if 1D, then first arg of box and res neglected)
            box - bounding box
            res - subdivisions of box in x (and y directions)
            diag - left or right (only for 2D mesh)
        """
        # get data
        vertex_to_coords, elt_to_vertex = regular_mesh_data(box=box, res=res, diag=diag)
        boundary = regular_mesh_bdata(res=res)

        dim = len(box)
        super(RegularMesh, self).__init__(vertex_to_coords, elt_to_vertex, boundary)
        self.__box = box
        self.__res = res
        self.__diag = diag

    def mesh_size(self):
        """ return h - size of elements in mesh """
        if self.dim() == 2:
            return max((self.__box[0][1] - self.__box[0][0])/float(self.__res[0]),\
                       (self.__box[1][1] - self.__box[1][0])/float(self.__res[1]))
        else:
            return (self.__box[0][1] - self.__box[0][0])/float(self.__res[0])


    def get_lagrange_data(self, deg=1):
        """ get Lagrange element data
        Input:
            deg - degree of Lagrange basis functions
        Returns:
            vertex_to_coords - node number to coordinate matrix
            elt_to_vertex - elt number to node numbers matrix
        """
        # check valid degree
        assert isinstance(deg, int) and 0 <= deg and deg <= 3, 'Invalid degree. Lagrange data only for deg=0,1,2,3.'

        # get data
        if deg == 0:
            return self.elt_to_ccoords(), np.array([[n] for n in range(self.n_elts())])
        elif deg == 1:
            return self._SuperMesh__vertex_to_coords, self._SuperMesh__elt_to_vertex
        else: # deg is 2 or 3
            return regular_mesh_data(self.__box, self.__res, deg=deg, diag=self.__diag)

    def get_bdata(self, deg=1):
        """ get Lagrange element boundary data
        Input:
            deg - degree of Lagrange basis functions
        Returns:
            boundary - dictionary of edge_to_nodes """

        # check valid degree
        assert isinstance(deg, int) and 0 <= deg and deg <= 3, 'Invalid degree. Lagrange data only for deg=0,1,2,3.'

        # get boundary data
        if deg == 0:
            if self.dim() == 2:
                x,y = self.__box; nx,ny = self.__res

                # assemble boundary groups
                bottom = []; top = []; left = []; right = []
                for n in range(nx):
                    bottom.append([2*n])
                    top.append([2*(nx*ny-n)-1])

                for n in range(ny):
                    left.insert(0,[2*n*nx+1])
                    right.append([2*(nx*(n+1)-1)])

                return {'bottom':bottom, 'top':top, 'right':right, 'left':left}
            else: # dim = 1
                # assemble boundary groups
                return {'left':[[0]], 'right':[[self.n_elts()-1]]}
        elif deg == 1:
            return self._SuperMesh__boundary
        else: # 2 or 3
            return regular_mesh_bdata(self.__res, deg)

#--------------------------------------------------------------------------------------#

class RectangleMesh(RegularMesh):
    """ rectangle mesh """
    def __init__(self, x=[0,1], y=[0,1], nx=4, ny=4, diag='right'):
        """
        Input:
            x,y - intervals defining bounding box
            nx,ny - subdivisions in x and y directions
        """
        super(RectangleMesh, self).__init__(box=[x,y], res=[nx,ny], diag=diag)

#--------------------------------------------------------------------------------------#

class UnitSquareMesh(RegularMesh):
    """ unit square mesh """
    def __init__(self, nx=4, ny=4, diag='right'):
        """
        Input:
            nx,ny - subdivisions in x and y directions
            diag - left or right
        """
        super(UnitSquareMesh, self).__init__(box=[[0,1],[0,1]], res=[nx,ny], diag=diag)

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':

    # mesh
    mesh = RectangleMesh(x=[0,10],y=[0,10],nx=10,ny=10)
    vertex_to_coords, elt_to_vertex = mesh.get_lagrange_data(deg=3)
    mesh.plot(dofs=vertex_to_coords)
