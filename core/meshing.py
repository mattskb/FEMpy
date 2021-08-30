import numpy as np
import math
from lagrange_data import regular_mesh_data, triangle_mesh_data

#--------------------------------------------------------------------------------------#
"""
def regular_mesh_data(box, res, diag):
    Data for regular triangular mesh
    Input:
        box                 - bounding box ([start_x,end_x],[start_y,end_y])
        res                 - number of divisions of x and y (nx,ny)
        diag                - diagonal to the left or right 
    Output:
        vertex_to_coords    - vertex number to coordinates
        elt_to_vertex       - element number to vertex numbers 
    

    # check valid data
    assert len(box) == len(res), 'Incompatible box and res arguments.'
    if len(box)!= 2:
        raise ValueError('Invalid box and res arguments.')

    # check valid diagonal arg
    diag = diag.lower()
    assert diag in {'left','l','right','r'}, 'Invalid diagonal argument.'
    
    x = box[0]; y = box[1]                      # bounding box
    nx = res[0]; ny = res[1]                    # number of cells in each direction (2 elts per cell)
    nx_vertex = nx+1; ny_vertex = ny+1          # number of vertex in x and y directions
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
"""

def plot_mesh(vertex_to_coords, elt_to_vertex, dof_to_coords, bdofs, title, figsize, grid, savefig):
    """Plots trianglular mesh with or without DOFs indicated
    Input:
        vertex_to_coords 	- vertex number to coordinates 
        elt_to_vertex 	 	- element number to vertex number
        dof_to_coords 		- DOF number to coordinate 
        bdofs               - numbers of boundary DOFS  
        title               - title of figure 
        figsize	    		- size of figure 
        grid                - True if grid lines are shown 
        savefig 	    	- file name in case of saved figure 
    """
    import matplotlib.pyplot as plt

    # initialize figure
    plt.figure(figsize=figsize)

    # plot mesh
    x = vertex_to_coords[:,0]; y = vertex_to_coords[:,1]
    plt.gca().set_aspect("equal")
    plt.triplot(x, y, elt_to_vertex, "g-", linewidth=2)

    # plot DOFs
    if dof_to_coords is not None:
        idofs = np.setdiff1d(range(dof_to_coords.shape[0]),bdofs)

        dof_x = dof_to_coords[idofs,0]; dof_y = dof_to_coords[idofs,1]
        plt.plot(dof_x,dof_y,"go",markersize=10)
    
    if bdofs is not None:
        dof_x = dof_to_coords[bdofs,0]; dof_y = dof_to_coords[bdofs,1]
        plt.plot(dof_x,dof_y,"bo",markersize=10)

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

    # gridlines
    plt.grid(grid)


    # save figure
    if savefig is not None:
        plt.savefig(savefig)

    plt.show()

#--------------------------------------------------------------------------------------#

class Mesh:
    """Triangular mesh 
    Input:
            vertex_to_coords         - vertex number to coordinates, i.e., np.array([[x, y]], float) 
            elt_to_vertex            - element number to vertex numbers, i.e., np.array([[v0, v1, v2]], int) 
            dof_data, optional e.g.,
                dof_to_coords        - DOF number to coordinates, i.e., np.array([[x, y]], float) 
                elt_to_dofs          - element number to DOF numbers, i.e., np.array([[dof_1,...,dof_n]], int)
    """
    def __init__(self, vertex_to_coords, elt_to_vertex):
        # set data
        self.__vertex_to_coords = vertex_to_coords
        self.__elt_to_vertex = elt_to_vertex
        self.__dof_data = {}
        self.__boundary_data = {}

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

    def set_dof_data(self, **dof_data):
        """ set DOF data given by keyword arguments """
        for k in dof_data.keys():
            self.__dof_data[k] = dof_data[k]

    def dof_data(self):
        """ print dof data and return list of keys"""
        if self.__dof_data:
            print("DOF data:")
            template = "\t{0:15}{1:20}"
            print(template.format("[name]","[shape]"))
            for k in self.__dof_data.keys():
                print(template.format(k, str(self.__dof_data[k].shape)))
            return self.__dof_data.keys()
        else: 
            print("No additional DOF data specified.")

    def n_dofs(self):
        """ get total number of DOFs in mesh """
        if self.__dof_data:
            return self.__dof_data["dof_to_coords"].shape[0]
        else:
            return self.n_vertices()

    def dof_to_coords(self, n=None):
        """Get DOF number to coordinates (global or of DOF n) """
        if self.__dof_data:
            if n is None:
                return self.__dof_data["dof_to_coords"]
            else:
                return self.__dof_data["dof_to_coords"][n,:]
        else:
            return self.vertex_to_coords(n)

    def elt_to_dofs(self, n=None):
        """ get elt number to DOF numbers (global or of elt n) """
        if self.__dof_data:
            if n is None:
                return self.__dof_data["elt_to_dofs"]
            else:
                return self.__dof_data["elt_to_dofs"][n,:]
        else:
            return self.elt_to_vertex(n)

    def elt_to_dofcoords(self, n=None):
        """return element number to DOF coords, or DOF coords of element n 
        i.e., n_eltsxn_dofs_eltx2 or n_elt_dofsx2 array """
        if n is None:
            return self.dof_to_coords()[self.elt_to_dofs()]
        else:
            return self.dof_to_coords()[self.elt_to_dofs(n)]

    def set_boundary_data(self, **boundary_data):
        """ Set boundary data given by keyword arguments """
        for k in boundary_data.keys():
            self.__boundary_data[k] = boundary_data[k]

    def boundary_groups(self):
        """ print boundary groups """
        if self.__boundary_data:
            print("Boundary edge-to-dofnumber groups:")
            template = "\t{0:10}{1:20}"
            print(template.format("[name]","[shape]"))
            for k in self.__boundary_data.keys():
                print(template.format(k, str(self.__boundary_data[k].shape)))
            return self.__boundary_data.keys()
        else:
            print("No boundary data specified.")

    def bedge_to_dofs(self, *group):
        """ Get all boundary edges to DOF numbers or only for group(s) specified by names """
        if len(group) == 0:
            return np.concatenate(tuple(self.__boundary_data.values()))
        else:
            inv_keys = [k for k in group if k not in self.__boundary_data.keys()]
            group = list(set(group) - set(inv_keys))
            if len(inv_keys) > 0:
                print("\nInvalid boundary group(s) given:")
                for k in inv_keys:
                    print("\t-", k)
            if len(group) > 0:
                return np.concatenate(tuple([self.__boundary_data[g] for g in group]))
            else:
                return None
        
    def plot(self, **kwargs):
        """Plots trianglular mesh with or without DOFs indicated
        Input (as keyword arguments):
            dof_to_coords       - DOF number to coordinate (Bool or as coordinates)
            title               - title of figure (default is generic title)
            figsize             - size of figure (default is (10,10))
            grid                - True if grid lines are shown (default: False)
            savefig             - file name in case of saved figure 
            boundary            - True if boundary DOFs indicated, can also be one or 
                                    more names of boundary groups (names in list if more than one)
        """
        dof_to_coords = kwargs.pop("dof_to_coords",None)
        title = kwargs.pop("title",None)
        figsize = kwargs.pop("figsize",(10,10))
        grid = kwargs.pop("grid",False)
        savefig = kwargs.pop("savefig",None)
        boundary = kwargs.pop("boundary",None)

        """
        if dof_to_coords in {1, True}:
            dof_to_coords = self.dof_to_coords()
        elif dof_to_coords in {0, False}:
            dof_to_coords = None
        """

        if boundary is not None:
            if isinstance(boundary, list):
                bedge_to_dofs = self.bedge_to_dofs(*boundary)
            elif boundary in {1, True}:
                bedge_to_dofs = self.bedge_to_dofs()
            else:
                bedge_to_dofs = self.bedge_to_dofs(boundary)
            boundary = np.unique(np.ravel(bedge_to_dofs)) if bedge_to_dofs is not None else None

        plot_mesh(self.vertex_to_coords(), self.elt_to_vertex(),
            dof_to_coords, boundary,
            title, figsize, grid, savefig)

#--------------------------------------------------------------------------------------#

class RectangleMesh(Mesh):
    """Rectangle shaped triangular mesh (first order unit square is default) 
    Input: 
            x,y             - intervals defining rectangle box
            nx,ny           - subdivisions in x and y directions
            deg             - degree of lagrange DOFs
            diag            - left or right 
    """
    def __init__(self, x=[0,1], y=[0,1], nx=4, ny=4, deg=1, diag='right'):
        vertex_to_coords, elt_to_vertex, boundary = regular_mesh_data((x, y), (nx, ny), 1, diag)
        super(RectangleMesh, self).__init__(vertex_to_coords, elt_to_vertex)

        if 1 < deg <= 10: 
            dof_to_coords, elt_to_dofs, boundary = regular_mesh_data((x, y), (nx, ny), deg, diag)
            self.set_dof_data(dof_to_coords=dof_to_coords, 
                              elt_to_dofs=elt_to_dofs)
        elif deg > 10:
            raise ValueError("Illegal value for degree (must be in range [1,10]).")

        self.set_boundary_data(top=boundary["top"],
                               bottom=boundary["bottom"],
                               left=boundary["left"],
                               right=boundary["right"])

        self.__box = (x, y)
        self.__res = (nx, ny)
        self.__deg = deg
        self.__diag = diag


    def deg(self):
        """ returns regree of lagrange DOFs """
        return self.__deg    

    def mesh_size(self):
        """ return h - size of elements in mesh """
        return max((self.__box[0][1] - self.__box[0][0])/float(self.__res[0]),\
                   (self.__box[1][1] - self.__box[1][0])/float(self.__res[1]))

#--------------------------------------------------------------------------------------#

class TriangleMesh(Mesh):
    """Regular mesh of reference triangle for plotting shape functions """
    def __init__(self, nx=4, ny=4):
        vertex_to_coords, elt_to_vertex = triangle_mesh_data(nx, ny)
        super(TriangleMesh, self).__init__(np.array(vertex_to_coords), np.array(elt_to_vertex))

#-------------------------------------------------------------------------------------#

if __name__ == '__main__':
    """ Here we do some tests """

    def rectangle_mesh_test(nx=3, ny=3, diag="r", deg=2):
        print("Test methods for Rectangle mesh (unit square) with params:\
            \n\tres: {}, diag: {}, deg: {}\n".format((nx,ny), diag, deg))

        print("\nBasic stuff -------------------------")
        mesh = RectangleMesh(nx=nx,ny=ny,deg=deg,diag=diag)
        print("- n elts: ", mesh.n_elts())
        print("- n vertices: ", mesh.n_vertices())
        print("- n edges: ", mesh.n_edges())
        print("- vertex_to_coords:", mesh.vertex_to_coords().shape,
            "\n  vertex 0:", mesh.vertex_to_coords(0))
        print("- elt_to_vertex:", mesh.elt_to_vertex().shape,
            "\n  elt 0:", mesh.elt_to_vertex(0))
        print("- elt_to_vcoords:", mesh.elt_to_vcoords().shape,
            "\n  elt 0:", np.array2string(mesh.elt_to_vcoords(0), prefix="  elt 0: "))
        print("- elt_to_ccoords:", mesh.elt_to_ccoords().shape,
            "\n  elt 0:", mesh.elt_to_ccoords(0))

        print("\nDOF stuff --------------------------")
        print("- dof data:", mesh.dof_data())
        print("- n dofs:", mesh.n_dofs())
        print("- dof_to_coords:", mesh.dof_to_coords().shape,
            "\n  elt 0:", mesh.dof_to_coords(0))
        print("- elt_to_dofs:", mesh.elt_to_dofs().shape,
            "\n  elt 0:", mesh.elt_to_dofs(0))
        print("- elt_to_dofcoords:", mesh.elt_to_dofcoords().shape,
            "\n  elt 0:", np.array2string(mesh.elt_to_dofcoords(0), prefix="  elt 0: "))

        print("\nBoundary stuff --------------------")
        print("- boundary groups:", mesh.boundary_groups())
        print("- bedge_to_dofs: (all)", mesh.bedge_to_dofs().shape)
        print("- bedge_to_dofs: (some)", mesh.bedge_to_dofs("top","left").shape)


        mesh.plot(dof_to_coords=True,boundary=["left", "right"])

        #print(mesh.vertex_to_coords().shape)
        #print(mesh.elt_to_vertex())
        #mesh.plot()

        #dof_to_coords, elt_to_dofs = regular_mesh_dofs(([1,0],[1,0]), (2,2), 2, "right")

    def dof_test(n=1, diag="r", deg=1):
        """ plot element wise DOFs """

        mesh = RectangleMesh(nx=n,ny=n,deg=deg,diag=diag)

        for j in range(mesh.n_elts()):
            dfs = mesh.elt_to_dofcoords(j)
            for d in dfs:
                mesh.plot(dof_to_coords=np.array([d]))

            if j > 10:
                break

    def mesh_test(n=16, deg=1, diag="r"):
        """ mesh class test """
        rect_mesh = RectangleMesh(nx=n,ny=n,deg=deg,diag=diag)

        elt_to_vertex = rect_mesh.elt_to_vertex()
        vertex_to_coords = rect_mesh.vertex_to_coords()

        mesh = Mesh(vertex_to_coords, elt_to_vertex)
        mesh.deg()


    n = 10
    #rectangle_mesh_test(nx=n, ny=n, diag="r", deg=2)
    #dof_test(n=4, diag="l", deg=1)
    mesh_test()