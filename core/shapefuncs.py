import numpy as np
from numpy.linalg import inv

def get_psi(m):
    """Get monomial basis component
    Input:
        m = [k,l] 
    Output:
        psi(x,y) = x^k * y^l 
    """
    def psi(xi):
        """ xi - 2D local coord or nx2 array of coords """
        try:
            return (xi[:,0]**m[0])*(xi[:,1]**m[1])
        except IndexError:
            return (xi[0]**m[0])*(xi[1]**m[1])
    return psi

#--------------------------------------------------------------------------------------#

def get_dpsi(m):
    """Get gradient of monomial basis component
    Input:
        m = [k,l] 
    Output:
        nabla psi(x,y) = [kx^{k-1} * y^l, x^k * ly^{l-1}]  
    """
    def dpsi(xi):
        """ xi - 2D local coord or nx2 array of coords """
        try:
            if m[0] == 0:
                comp1 = 0*xi[:,0]
            else:
                comp1 = m[0]*(xi[:,0]**(m[0]-1))*(xi[:,1]**m[1])

            if m[1] == 0:
                comp2 = 0*xi[:,1]
            else:
                comp2 = m[1]*(xi[:,1]**(m[1]-1))*(xi[:,0]**m[0])
        except IndexError:
            if m[0] == 0:
                comp1 = 0
            else:
                comp1 = m[0]*(xi[0]**(m[0]-1))*(xi[1]**m[1])

            if m[1] == 0:
                comp2 = 0*xi[1]
            else:
                comp2 = m[1]*(xi[1]**(m[1]-1))*(xi[0]**m[0])
        return np.array([comp1, comp2]).T
    return dpsi

#--------------------------------------------------------------------------------------#

def monomial_basis(deg):
    """Get complete monomial basis and derivative
    Input:
        deg - degree monomial
    Output:
        [psi_1,...,psi_n], [dpsi_1,...,dpsi_n] 
    """
    psi = [get_psi([m,n]) for m in range(deg+1) for n in range(deg+1) if m+n<=deg]
    dpsi = [get_dpsi([m,n]) for m in range(deg+1) for n in range(deg+1) if m+n<=deg]
    return psi, dpsi

#--------------------------------------------------------------------------------------#

def get_lagrange_nodes(deg):
    """Get nodal points in ref elt for Lagrange basis
    Input:
        deg - degree of polynomial space (number of nodes)
    Output: 
        [x,y] coordinates of nodes
    """
    pts = np.linspace(0,1,deg+1)
    return np.array([[xi,yi] for yi in pts for xi in pts if xi+yi<=1])


#--------------------------------------------------------------------------------------#

def get_phi(i, coeffs, psi, n_nodes):
    """Get i'th Lagrange basis function or derivative
    Input:
        coeffs      - coefficient matrix for expansion in monomial basis
        psi         - complete monomial basis
        n_nodes     - number of nodes 
    """
    def phi(xi):
        """ xi - 2D local coord or nx2 array of coords """
        return np.sum([coeffs[i,j]*psi[j](xi) for j in range(n_nodes)], axis=0)
    return phi

#--------------------------------------------------------------------------------------#

def get_dphi(i, coeffs, dpsi, n_nodes):
    """Get i'th Lagrange basis function derivative
    Input:
        coeffs      - coefficient matrix for expansion in monomial basis
        dpsi        - complete monomial basis derivative
        n_nodes     - number of nodes 
    """
    def dphi(xi):
        """ xi - 2D local coord or nx2 array of coords """
        try:
            comp1 = np.sum([coeffs[i,j]*dpsi[j](xi)[:,0] for j in range(n_nodes)], axis=0)
            comp2 = np.sum([coeffs[i,j]*dpsi[j](xi)[:,1] for j in range(n_nodes)], axis=0)
        except IndexError:
            comp1 = np.sum([coeffs[i,j]*dpsi[j](xi)[0] for j in range(n_nodes)], axis=0)
            comp2 = np.sum([coeffs[i,j]*dpsi[j](xi)[1] for j in range(n_nodes)], axis=0)
        return np.array([comp1, comp2]).T                      
    return dphi

#--------------------------------------------------------------------------------------#

def get_lagrange_basis(deg):
    """Get complete Lagrange basis and derivative, 
    where monomial coefficients are found using Vandermonde matrix, i.e.,

        [psi_j(x_i)]_ij * [a_ij]_ij = Id, thus, [a_ij]_ij = inv([psi_j(x_i)]_ij)

    Input:
        deg - degree of polynomial space
    Output:
        [phi_1,...,phi_n], [dphi_1,...,dphi_n] 
    """
    nodes = get_lagrange_nodes(deg)
    n_nodes = nodes.shape[0]
    psi, dpsi = monomial_basis(deg)

    # assmeble Vandermonde matrix
    V = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            V[i,j] = psi[j](nodes[i,:])
    monomial_coeffs = inv(V).T

    phi = [get_phi(i, monomial_coeffs, psi, n_nodes) for i in range(n_nodes)]
    dphi = [get_dphi(i, monomial_coeffs, dpsi, n_nodes) for i in range(n_nodes)]

    return phi, dphi

#--------------------------------------------------------------------------------------#

class ShapeFunctions:
    """  Lagrange shape functions defined on reference element """
    def __init__(self, deg):
        """
        deg - degree of shape functions (1,2 or 3)
        """

        # check valid dimentions and degree
        assert isinstance(deg, int) and (1 <= deg <= 3), 'Invalid degree for Lagrange shape functions.'

        # get shape functions
        phi, dphi = get_lagrange_basis(deg)

        self.__phi = phi
        self.__dphi = dphi

    def eval(self, n, xi, derivative=False):
        """ eval n'th shape function, either phi or dphi, at xi """
        if derivative:
            return self.__dphi[n](xi)
        else:
            return self.__phi[n](xi)

    def n_dofs(self):
        """ return number of DOFs """
        return len(self.__phi)

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
    """ Here we do some tests """

    
    def eval_test(x):
        """ print output of the functions """
        print("Input: ", x)

        print("\npsi eval test")
        psi = get_psi([1,1])
        print(psi(x))

        print("\ndpsi eval test")
        dpsi = get_dpsi([1,1])
        print(dpsi(x))

        deg = 1; n_nodes = 3
        psi, dpsi = monomial_basis(deg)
        print("\nphi test")
        coeffs = np.ones((3,3))
        phi = get_phi(0, coeffs, psi, n_nodes)
        print(phi(x))

        print("\ndphi test")
        dphi = get_dphi(0, coeffs, dpsi, n_nodes)
        print(dphi(x))

        print("\nlagrange nodes, deg ", deg)
        lnodes = get_lagrange_nodes(deg)
        print(lnodes)

    def plot_sfns(deg):
        """ plot the lagrange basis functions """
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation 
        from meshing import TriangleMesh

        # get shape functions
        sf = ShapeFunctions(deg)
        n_dofs = sf.n_dofs()

        # get mesh of ref triangle and make triangulation
        mesh = TriangleMesh(10,10)
        vertex_to_coords = mesh.vertex_to_coords()
        elt_to_vertex = mesh.elt_to_vertex()
        X = vertex_to_coords[:,0]; Y = vertex_to_coords[:,1]
        tri = Triangulation(X, Y, elt_to_vertex)

        # plot the basis functions
        for j in range(n_dofs):
            #fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10,10))
            #surf = ax.plot_trisurf(tri, sf.eval(j, vertex_to_coords), shade=1)

            fig, ax = plt.subplots(figsize=(10,10),num=j+1)
            cont = ax.tricontourf(tri, sf.eval(j, vertex_to_coords), 100)            
            plt.colorbar(cont)

            ax.set_title("Degree {} Lagrange basis function {}/{}.".format(deg, j+1, n_dofs))
            
        # plot the DOFs
        mesh.plot(dof_to_coords=get_lagrange_nodes(deg), 
            title="Reference triangle with Lagrange nodes")
            


    linsp = np.linspace(0,1,10)
    x = np.array([linsp, linsp]).T
    # x = np.array([1,2])
    #eval_test(x)

    deg = 1
    plot_sfns(deg)






    #sf = ShapeFunctions(1)
    #for n in range(3):
    #    print(sf.eval(n, [0,0], derivative=True))

    def inside(xi):
        """ true if xi is inside ref element """
        TOL = 1e-8
        return xi[0] >=0.-TOL and xi[1] >= 0.-TOL and xi[0] + xi[1] <= 1.+TOL

    
            

    def plot2D():
        """ plot 2D Lagrange shape functions on ref element """
        xi = np.linspace(0,1,101)
        yi = np.linspace(0,1,101)
        #temp = xi + yi > 1
        #xi[temp] = 0
        #yi[temp] = 0
        #Xi, Yi = np.meshgrid(xi, yi)

        p1 = ShapeFunctions(1)
        p2 = ShapeFunctions(2)
        p3 = ShapeFunctions(3)

        p11 = np.zeros((101,101))
        p12 = np.zeros((101,101))
        p13 = np.zeros((101,101))

        p21 = np.zeros((101,101))
        p22 = np.zeros((101,101))
        p23 = np.zeros((101,101))
        p24 = np.zeros((101,101))
        p25 = np.zeros((101,101))
        p26 = np.zeros((101,101))

        p31 = np.zeros((101,101))
        p32 = np.zeros((101,101))
        p33 = np.zeros((101,101))
        p34 = np.zeros((101,101))
        p35 = np.zeros((101,101))
        p36 = np.zeros((101,101))
        p37 = np.zeros((101,101))
        p38 = np.zeros((101,101))
        p39 = np.zeros((101,101))
        p310 = np.zeros((101,101))


        # x is columns, y is rows
        for i in range(101):
            for j in range(101):
                x = [xi[i],yi[j]]
                if inside(x):
                    p11[j,i] = p1.eval(0,x)
                    p12[j,i] = p1.eval(1,x)
                    p13[j,i] = p1.eval(2,x)

                    p21[j,i] = p2.eval(0,x)
                    p22[j,i] = p2.eval(1,x)
                    p23[j,i] = p2.eval(2,x)
                    p24[j,i] = p2.eval(3,x)
                    p25[j,i] = p2.eval(4,x)
                    p26[j,i] = p2.eval(5,x)

                    p31[j,i] = p3.eval(0,x)
                    p32[j,i] = p3.eval(1,x)
                    p33[j,i] = p3.eval(2,x)
                    p34[j,i] = p3.eval(3,x)
                    p35[j,i] = p3.eval(4,x)
                    p36[j,i] = p3.eval(5,x)
                    p37[j,i] = p3.eval(6,x)
                    p38[j,i] = p3.eval(7,x)
                    p39[j,i] = p3.eval(8,x)
                    p310[j,i] = p3.eval(9,x)

        # plot linear functions
        print(p11)
        plt.figure(num='2D linear shape functions', figsize=(6,7))
        plt.subplot(311)
        CS1 = plt.contourf(xi, yi, p11, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_1')
        plt.axis('off')

        plt.subplot(312)
        CS1 = plt.contourf(xi, yi, p12, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_2')
        plt.axis('off')

        plt.subplot(313)
        CS1 = plt.contourf(xi, yi, p13, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_3')
        plt.axis('off')

        # plot quadratic functions
        plt.figure(num='2D quadratic shape functions', figsize=(10,7.5))
        plt.subplot(321)
        CS1 = plt.contourf(xi, yi, p21, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_1')
        plt.axis('off')

        plt.subplot(322)
        CS1 = plt.contourf(xi, yi, p22, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_2')
        plt.axis('off')

        plt.subplot(323)
        CS1 = plt.contourf(xi, yi, p23, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_3')
        plt.axis('off')

        plt.subplot(324)
        CS1 = plt.contourf(xi, yi, p24, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_4')
        plt.axis('off')

        plt.subplot(325)
        CS1 = plt.contourf(xi, yi, p25, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_5')
        plt.axis('off')

        plt.subplot(326)
        CS1 = plt.contourf(xi, yi, p26, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_6')
        plt.axis('off')

        # plot cubic functions
        plt.figure(num='2D cubic shape functions', figsize=(14,7.5))
        plt.subplot(251)
        CS1 = plt.contourf(xi, yi, p31, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_1')
        plt.axis('off')

        plt.subplot(252)
        CS1 = plt.contourf(xi, yi, p32, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_2')
        plt.axis('off')

        plt.subplot(253)
        CS1 = plt.contourf(xi, yi, p33, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_3')
        plt.axis('off')

        plt.subplot(254)
        CS1 = plt.contourf(xi, yi, p34, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_4')
        plt.axis('off')

        plt.subplot(255)
        CS1 = plt.contourf(xi, yi, p35, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_5')
        plt.axis('off')

        plt.subplot(256)
        CS1 = plt.contourf(xi, yi, p36, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_6')
        plt.axis('off')

        plt.subplot(257)
        CS1 = plt.contourf(xi, yi, p37, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_7')
        plt.axis('off')

        plt.subplot(258)
        CS1 = plt.contourf(xi, yi, p38, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_8')
        plt.axis('off')

        plt.subplot(259)
        CS1 = plt.contourf(xi, yi, p39, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_9')
        plt.axis('off')

        plt.subplot(2, 5, 10)
        CS1 = plt.contourf(xi, yi, p310, 100)
        CB = plt.colorbar(CS1, shrink=0.8)
        plt.title('phi_10')
        plt.axis('off')

        plt.show()

    #plot2D()







