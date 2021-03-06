import numpy as np
import math

def plot_mesh(node_to_coords, elt_to_nodes, dim, dofs=None, file=None):
    """
    makes a plot of mesh defined by cell_coords and elt_to_nodes with or without node_to_coords
    Input:
        node_to_coords - node number to coordinates matrix
        elt_to_nodes - element number to node numbers matrix
        dim - spatial dimension
        dofs - DOF number to coordinate, if DOFs are shown in figure
        file - file name of figure is saved to .pdf file
    """
    import matplotlib.pyplot as plt

    if dofs is not None:
        plt.figure('Mesh plot with DOFs', figsize=(8,6))
    else:
        plt.figure('Mesh plot', figsize=(8,6))

    if dim == 2:
        # plot mesh
        x = node_to_coords[:,0]; y = node_to_coords[:,1]
        plt.gca().set_aspect('equal')
        plt.triplot(x, y, elt_to_nodes[:,:3], 'g-', linewidth=.5)

        # plot DOFs
        if dofs is not None:
            dof_x = dofs[:,0]; dof_y = dofs[:,1]
            plt.plot(dof_x,dof_y,'go')
            plt.title('Triangular mesh with DOFs.')
        else:
            plt.title('Triangular mesh.')

        plt.xlabel('x')
        plt.ylabel('y')
    else: # dim = 1
        # plot horizontal line
        xmin = np.min(node_to_coords); xmax = np.max(node_to_coords)
        plt.axhline(y=0, xmin=xmin, xmax=xmax, color='g', linewidth=2.)
        plt.ylim(-1,1)

        # add elt_to_nodes
        elt_to_vcoord = np.unique(node_to_coords[elt_to_nodes[:,:2]])
        for v in elt_to_vcoord:
            plt.axvline(v, color='k', linestyle='--', linewidth=0.5)
        plt.yticks([])

        if dofs is not None:
            y = np.zeros(len(dofs))
            plt.plot(dofs, y, 'go')
            plt.title('Interval mesh with DOFs.')
        else:
            plt.title('Interval mesh.')

    # save figure
    if file is not None:
        #plt.savefig(plotfile + '.png')
        plt.savefig(plotfile + '.pdf')

    plt.show()

#--------------------------------------------------------------------------------------#

def read_gmsh(file):
    """
    Reads mesh data from .msh Gmsh-file
    File must specify at least two physical groups; boundary and interior
    Three groups can also be specified, e.g. Dirichlet and Neumann boundary node_to_coords
    Output:
        node_to_coords - node number to coordinate
        elt_to_nodes - element number to node numbers
        boundary - dictionary of other groups defined in mesh i.e. boundary
    """
    with open(file) as fp:
        print('Read mesh file\n---------------------------------')
        print("Name:\n \t{}".format(fp.name))
        fp.readline() # '$MeshFormat\n'
        fp.readline() # version no.\n
        fp.readline() # '$EndMeshFormat\n'

        assert fp.readline() == '$PhysicalNames\n', 'File not read correctly.'
        n_phys_groups = int(fp.readline())

        names = []; dims = []
        print('Physical groups:')
        for n in range(n_phys_groups):
            dim, num, name = fp.readline().split()
            print('\t{}, n={}, d={}'.format(name, num, dim))
            names.append(name[1:-1])
            dims.append(int(dim))
        assert fp.readline() == '$EndPhysicalNames\n', 'File not read correctly.'
        print('---------------------------------')

        max_dim_name = names[dims.index(max(dims))] # name of group for elt_to_nodes of mesh

        # node_to_coords
        assert fp.readline() == '$Nodes\n', 'File not read correctly.'
        n_nodes = int(fp.readline())
        node_to_coords = np.fromfile(fp,dtype=float,count=n_nodes*4,sep=" ").reshape((n_nodes,4))
        node_to_coords = node_to_coords[:,1:3] # need only x and y coord
        assert len(node_to_coords) == n_nodes, 'Not all node_to_coords included.'
        assert fp.readline() == '$EndNodes\n', 'File not read correctly.'

        # elt_to_nodes
        assert fp.readline() == '$Elements\n', 'File not read correctly.'
        n_elts = int(fp.readline())

        # boundary groups
        boundary = {name : [] for name in names}
        for n in range(n_elts):
            words = fp.readline().split()
            group_tag = int(words[3])-1
            n_tags = int(words[2])
            local_nodes = [int(words[i])-1 for i in range(3+n_tags,len(words))]
            boundary[names[group_tag]].append(local_nodes)

        assert fp.readline() == '$EndElements\n', 'File not read correctly.'
        assert sum([len(boundary[name]) for name in names]) == n_elts, 'Not all elements included.'

        # extract group with highest dimension and delete from dictionary
        elt_to_nodes = np.array(boundary[max_dim_name], dtype=int)
        del boundary[max_dim_name]

    return node_to_coords, elt_to_nodes, boundary

#--------------------------------------------------------------------------------------#

def regular_mesh_data(box=[[0,1],[0,1]], res=[4,4], deg=1, diag='right'):
    """ assembles lagrange data for regular mesh (interval in 1D or triangular in 2D)
    Input:
        box - bounding box
        res - number of divisions of x (and y intervals)
        deg - degree of Langrange basis functions (1 through 3)
        diag - diagonal to the left or right (only for 2D data)
    Output:
        node_to_coords - node number to coordinate matrix
        elt_to_nodes - element number to node numbers matrix
    """

    # check valid data
    assert len(box) == len(res), 'Incompatible box and res arguments.'

    if len(box) == 1: # <-------------- 1D uniform interval mesh
        # data
        x = box[0]          # interval
        n = res[0]          # number of divisions of interval
        n_nodes = deg*n+1   # number of nodes

        # assemble node_to_coords
        temp = np.linspace(x[0],x[1],n_nodes)
        node_to_coords = []
        for node in temp:
            node_to_coords.append([node,0])
        node_to_coords = np.array(node_to_coords, dtype=float)

        # assemble elt_to_nodes
        elt_to_nodes = []
        for i in range(n):
            v0 = i*deg; v1 = v0+1; v2 = v1+1; v3 = v2+1
            if deg == 1:
                elt_to_nodes.append([v0, v1])
            elif deg ==2:
                elt_to_nodes.append([v0, v2, v1])
            elif deg == 3:
                elt_to_nodes.append([v0, v3, v1, v2])

        elt_to_nodes = np.array(elt_to_nodes, dtype=int)

    elif len(box) == 2: # <-------------- 2D uniform triangular mesh
        # check valid diagonal
        diag = diag.lower()
        assert diag == 'right' or diag == 'left', 'Invalid diagonal argument.'

        # data
        x = box[0]; y = box[1]                      # bounding box
        nx = res[0]; ny = res[1]                    # number of divisions of box
        nx_nodes = deg*nx+1; ny_nodes = deg*ny+1    # number of nodes in x and y directions
        n_nodes = nx_nodes*ny_nodes                 # total number of nodes
        n_elts = 2*nx*ny                            # number of elements

        # assemble node_to_coords
        xx = np.linspace(x[0],x[1],nx_nodes)
        yy = np.linspace(y[0],y[1],ny_nodes)
        node_to_coords = []
        for iy in range(ny_nodes):
            for ix in range(nx_nodes):
                node_to_coords.append([xx[ix], yy[iy]])
        node_to_coords = np.array(node_to_coords, dtype=float)
        assert len(node_to_coords) == n_nodes, 'Assembly of node_to_coords failed.'

        # assemble elt_to_nodes, anti-clockwise numbering of nodes
        elt_to_nodes = []
        if deg == 1:
            for iy in range(ny):
                for ix in range(nx):
                    v0 = iy*nx_nodes+ix; v1 = v0+1
                    v2 = v0+nx_nodes; v3 = v2+1

                    if diag == 'right':
                        elt_to_nodes.append([v0, v1, v3])
                        elt_to_nodes.append([v0, v3, v2])
                    else:
                        elt_to_nodes.append([v0, v1, v2])
                        elt_to_nodes.append([v1, v3, v2])
        elif deg == 2:
            for iy in range(ny):
                for ix in range(nx):
                    v0 = (2*iy)*nx_nodes+(2*ix); v1 = v0+1; v2 = v1+1
                    v3 = v0+nx_nodes; v4 = v3+1; v5 = v4+1
                    v6 = v3+nx_nodes; v7 = v6+1; v8 = v7+1

                    if diag == 'right':
                        elt_to_nodes.append([v0, v2, v8, v1, v5, v4])
                        elt_to_nodes.append([v0, v8, v6, v4, v7, v3])
                    else:
                        elt_to_nodes.append([v0, v2, v6, v1, v4, v3])
                        elt_to_nodes.append([v2, v8, v6, v5, v7, v4])
        elif deg == 3:
            for iy in range(ny):
                for ix in range(nx):
                    v0 = (3*iy)*nx_nodes+(3*ix); v1 = v0+1; v2 = v1+1; v3 = v2+1
                    v4 = v0+nx_nodes; v5 = v4+1; v6=v5+1; v7=v6+1
                    v8 = v4+nx_nodes; v9 = v8+1; v10 = v9+1; v11 = v10+1
                    v12 = v8+nx_nodes; v13 = v12+1; v14 = v13+1; v15 = v14+1

                    if diag == 'right':
                        elt_to_nodes.append([v0, v3, v15, v1, v2, v7, v11, v10, v5, v6])
                        elt_to_nodes.append([v0, v15, v12, v5, v10, v14, v13, v8, v4, v9])
                    else:
                        elt_to_nodes.append([v0, v3, v12, v1, v2, v6, v9, v8, v4, v5])
                        elt_to_nodes.append([v3, v15, v12, v7, v11, v14, v13, v9, v6, v10])


        elt_to_nodes = np.array(elt_to_nodes, dtype=int)
        assert len(elt_to_nodes) == n_elts, 'Assembly of elt_to_nodes failed.'

    else:
        raise ValueError('Invalid box and res arguments.')

    return node_to_coords, elt_to_nodes

#--------------------------------------------------------------------------------------#

def regular_mesh_bdata(res=[4,4], deg=1):
    """ assembles boundary data for regular mesh (interval in 1D or triangular in 2D)
    Input:
        res - number of divisions of x (and y intervals)
        deg - degree of Langrange basis functions (1 through 3)
    Output:
        boundary - dictionary of edge number to node numbers matrices
    """

    # assemble boundary
    if len(res) == 1: # <-------------- 1D uniform interval mesh
        n = res[0]                                  # number of divisions of interval
        n_nodes = deg*n+1                           # number of node_to_coords

        boundary = {'left': [[0]], 'right': [[n_nodes-1]]}

    elif len(res) == 2: # <-------------- 2D uniform triangular mesh
        nx = res[0]; ny = res[1]                    # number of divisions of interval
        n_bedges = 2*nx + 2*ny                      # number of boundary edges
        nx_nodes = deg*nx+1; ny_nodes = deg*ny+1    # number of nodes in x and y directions

        bottom = []; top = []; right = []; left = []
        # assemble edge_to_nodes
        for n in range(nx):
            n *= deg
            bnodes = [n+i for i in range(deg+1)]
            bottom.append([bnodes[0], bnodes[-1]] + [bnodes[i] for i in range(1,deg)])

            tnodes = [nx_nodes*ny_nodes - (n+i) for i in range(1,deg+2)]
            #top.append([tnodes[0], tnodes[-1]] + [tnodes[i] for i in range(1,deg)])
            top.append([tnodes[-1], tnodes[0]] + [tnodes[i] for i in range(1,deg)])

        for n in range(ny):
            rnodes = [(n*deg+i)*nx_nodes-1 for i in range(1,deg+2)]
            right.append([rnodes[0], rnodes[-1]] + [rnodes[i] for i in range(1,len(rnodes)-1)])

            lnodes = [(n*deg+i)*nx_nodes for i in range(deg+1)]
            left.append([lnodes[-1], lnodes[0]] + [lnodes[i] for i in range(1,len(lnodes)-1)])

        top.reverse(); #left.reverse()
        boundary = {'bottom':bottom, 'top':top, 'right':right, 'left':left}

    else:
        raise ValueError('Invalid res argument.')

    return boundary

#--------------------------------------------------------------------------------------#

def get_rt_data(elt_to_nodes):
    """ generates edges of triangulation def by elts_to_nodes (from Anjam)
    Input:
        elt_to_nodes - elt number to node numbers matrix
    Output:
        edge_to_nodes - edge number to node numbers matrix
        elt_to_edges - elt number to edge numbers matrix
        elt_to_signs - elt number to signs of edges
     """
    n_elts = elt_to_nodes.shape[0]

    # extract sets of edges
    e1 = elt_to_nodes[:,[1,2]]
    e2 = elt_to_nodes[:,[2,0]]
    e3 = elt_to_nodes[:,[0,1]]

    edge_to_nodes = np.zeros((n_elts*3, 2), dtype=int)
    for j in range(n_elts):
        i = j*3
        edge_to_nodes[i,:] = e1[j,:]
        edge_to_nodes[i+1,:] = e2[j,:]
        edge_to_nodes[i+2,:] = e3[j,:]

    edge_to_nodes = np.sort(edge_to_nodes,1)
    temp = []; J = []; I = []; i = 0
    for j in range(3*n_elts):
        e = tuple(edge_to_nodes[j])
        if e not in temp:
            temp.append(e)
            J.append(j)
            I.append(i); i+=1
        else:
            k = temp.index(e)
            I.append(k)

    #print np.allclose(edge_to_nodes - np.array(temp)[I,:],0)
    #print np.allclose(edge_to_nodes[J,:] - np.array(temp),0)
    edge_to_nodes = edge_to_nodes[J,:]
    elt_to_edges = np.array(I).reshape(n_elts,3)

    # get signs
    tmp = elt_to_nodes[:,[1,2,0]] - elt_to_nodes[:,[2,0,1]]
    elt_to_signs = np.divide(tmp, np.abs(tmp))
    return edge_to_nodes, elt_to_edges, elt_to_signs

#--------------------------------------------------------------------------------------#

class SuperMesh(object):
    """ super class for meshes """
    def __init__(self, node_to_coords, elt_to_nodes, dim, boundary=None):
        """
        Input:
            node_to_coords - node number to coordinates matrix
            elt_to_nodes - elt number to node numbers matrix
            boundary - dictionary of boundary edge to nodes matrices (optional)
            dim - spatial dimension of mesh
        """

        elt_to_vcoords = node_to_coords[elt_to_nodes[:,:dim+1]]         # elt number to vertex coords
        n_elts = elt_to_nodes.shape[0]                                  # number of elements
        n_vertices = node_to_coords.shape[0]                            # number of vertices
        n_edges = n_elts + n_vertices - 1                               # number of edges (Eulers formula for 2D)

        # set data
        self.__dim = dim
        self.__node_to_coords = node_to_coords
        self.__elt_to_nodes = elt_to_nodes
        self.__elt_to_vcoords = elt_to_vcoords
        self.__boundary = boundary
        self.__n_elts = n_elts
        self.__n_vertices = n_vertices
        self.__n_edges = n_edges

    def dim(self):
        """ return spatial dimension of mesh """
        return self.__dim

    def n_elts(self):
        """ return number of elements in mesh """
        return self.__n_elts

    def n_vertices(self):
        """ return number of vertices in mesh """
        return self.__n_vertices

    def n_edges(self):
        """ return number of edges (only for 2D mesh) """
        return self.__n_edges

    def node_to_coords(self):
        """ get node number to coordinate matrix """
        return self.__node_to_coords

    def elt_to_nodes(self):
        """ get elt number to node numbers matrix """
        return self.__elt_to_nodes

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

    def get_rt_data(self, deg=0):
        """ get Raviart-Thomas element data (only degree 0 available)
        Output:
            edge_to_nodes - edge number to node numbers matrix
            elt_to_edges - elt number to edge numbers matrix
            elt_to_signs - elt number to edge signs matrix
        """
        assert self.__dim == 2, 'Raviart-Thomas data only for 2D mesh.'
        if deg == 0:
            return get_rt_data(self.__elt_to_nodes)
        else:
            raise ValueError('Invalid degree. Raviart-Thomas only implemented for deg = 0.')

    def plot(self, dofs=None, file=None):
        """ plot figure of mesh
        DOFs given by dofs can be shown in figure
        file - name of .pdf file if figure is saved
        """
        plot_mesh(self.__node_to_coords, self.__elt_to_nodes, self.dim(), dofs=dofs, file=file)

#--------------------------------------------------------------------------------------#

class Gmesh(SuperMesh):
    """ 2D mesh constructed from .msh Gmsh-file"""
    def __init__(self, file):
        """
        Input:
            file - .msh Gmsh-file defining node_to_coords, elt_to_nodes and boundary
        """
        node_to_coords, elt_to_nodes, boundary = read_gmsh(file); dim = 2
        super(Gmesh, self).__init__(node_to_coords, elt_to_nodes, dim, boundary)
        n_elt_nodes = elt_to_nodes.shape[1]
        self.__deg = int( (math.sqrt(9-8*(1-n_elt_nodes))-3)/2 )

    def deg(self):
        """ return degree of Lagrange basis functions """
        return self.__deg

    def get_lagrange_data(self, deg=1):
        """ get Lagrange element data
        Output:
            node number to coords matrix
            element number to node numbers matrix
        """
        if deg == 0:
            return self.elt_to_ccoords(), np.array([[n] for n in range(self.n_elts())])
        elif deg == self.deg():
            return self._SuperMesh__node_to_coords, self._SuperMesh__elt_to_nodes
        else:
            raise ValueError('Invalid degree. Only deg=0 and deg={} available.'.format(self.deg()))

    def get_bdata(self, deg=1):
        """ return dictionary of boundary edge to node numbers matrices """
        if deg == self.deg():
            return self._SuperMesh__boundary
        else:
            raise ValueError('Invalid degree. Degree of this mesh is {}.'.format(self.deg()))

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
        node_to_coords, elt_to_nodes = regular_mesh_data(box=box, res=res, diag=diag)
        boundary = regular_mesh_bdata(res=res)

        dim = len(box)
        super(RegularMesh, self).__init__(node_to_coords, elt_to_nodes, dim, boundary)
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
            node_to_coords - node number to coordinate matrix
            elt_to_nodes - elt number to node numbers matrix
        """
        # check valid degree
        assert isinstance(deg, int) and 0 <= deg and deg <= 3, 'Invalid degree. Lagrange data only for deg=0,1,2,3.'

        # get data
        if deg == 0:
            return self.elt_to_ccoords(), np.array([[n] for n in range(self.n_elts())])
        elif deg == 1:
            return self._SuperMesh__node_to_coords, self._SuperMesh__elt_to_nodes
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

class IntervalMesh(RegularMesh):
    """ regular interval mesh """
    def __init__(x=[0,1], n=10):
        """
        Input:
            x - interval
            n - number of subdivisions of interval
        """
        super(RegularMesh, self).__init__(box=[x], res=[n])

#--------------------------------------------------------------------------------------#

class UnitIntMesh(RegularMesh):
    """ unit interval mesh """
    def __init__(self, n=10):
        """
        Input:
            n - number of divisions of unit interval
        """
        super(UnitIntMesh, self).__init__(box=[[0,1]], res=[n])


if __name__ == '__main__':

    # mesh
    mesh = RectangleMesh(x=[0,10],y=[0,10],nx=100,ny=100)
    mesh.plot()







