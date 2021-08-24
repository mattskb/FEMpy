import numpy as np
from finiteElement import Element

class Space:
    """Function space class
    Input:
        mesh            - mesh it is built on
        fe              - Finite Element 
        gauss           - degree of Gaussian quadrature (optional, default is 4)
    """
    def __init__(self, mesh, deg, gauss=4):
        # set data
        self.__gauss = gauss                
        self.__mesh = mesh                            
        self.__deg = deg
        self.__fe = Element(deg, gauss)       

        ndofs = mesh.dof_to_coords()                         

    def n_dofs(self):
        """ return total number of DOFs """
        return self.__mesh.n_dofs()

    def elt_to_dofs(self):
        """ return element number to DOF numbers """
        return self.__elt_to_dofs

    def assemble(self, c=None, derivative=False, cond=False):
        """Assemble linear system
        Input:
            c           - coefficient (variable or constant), can be tensor valued if appropriate
            derivative  - True if derivative of shape function
            cond        - True if condition number is calculated and printed
        Output:
            A           - stiffness/mass matrix (dense)
        """

        # assemble
        #print "Assembling linear system..."
        A = np.zeros((self.n_dofs(), self.n_dofs()))                  # initialize global assembly matrix

        for j in range(self.__mesh.n_elts()):
            # local assemble for element j
            self.__fe.set_data(self.__mesh.elt_to_vcoords(j),\
            self.__mesh.elt_to_dofcoords(j))                          # give element data to FE
            elt_assemble = self.__fe.assemble(c, derivative)          # element assembly
            #print(elt_assemble)
            #from numpy.linalg import cond
            #print(cond(elt_assemble))
            #print("========================================")
            local_dofs = np.array([self.__mesh.elt_to_dofs(j)])       # indices
            A[local_dofs.T, local_dofs] += elt_assemble               # add to global assembly

        if cond:
            from numpy.linalg import cond
            print("Condition number for assembly of {} matrix: {}".format("stiffness" if derivative else "mass", cond(A)))

        return A


    def stiffness(self, c=None, cond=False):
        """ assemble stiffness matrix """
        return self.assemble(c, True, cond)

    def mass(self, c=None, cond=False):
        """ assemble mass matrix """
        return self.assemble(c, False, cond)

    def rhs(self, f):
        """Assemble right hand side vector 
        Input:
            f - function (or constant)
        """
        rhs = np.zeros(self.n_dofs())
        if f == 0:
            return rhs
        else:
            # assemble
            for j in range(self.__mesh.n_elts()):
                self.__fe.set_data(self.__mesh.elt_to_vcoords(j),\
                    self.__mesh.elt_to_dofcoords(j))                          # give element data to FE

                local_dofs = np.array([self.__mesh.elt_to_dofs(j)])
                #print(np.array([self.__mesh.elt_to_vertices(j)]))
                #print(self.__fe.assemble_rhs(f))

                rhs[local_dofs] += self.__fe.assemble_rhs(f)
            return rhs

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
    import math
    from meshing import RectangleMesh


    deg = 1
    mesh = RectangleMesh(nx=1,ny=1,diag="l")
    fs = Space(mesh, deg)
    A = fs.stiffness(cond=True)
    #print(A)
    #print("no of dofs: ", fs.n_dofs())

    def dirichlet_ex(n=16, deg=1, gauss=4, diag='right', plot=True):
        """ Poisson w/ homogenous Dirichlet bc in 2D
        """
        from scipy.sparse import issparse, csc_matrix, csr_matrix, dia_matrix
        import scipy.sparse.linalg as spla
        import matplotlib.pyplot as plt

        print("\nDirichlet in 2D")
        # data
        mesh = RectangleMesh(nx=n,ny=n,deg=1,diag=diag)

        f = lambda x: 32.*(x[:,1]*(1.-x[:,1]) + x[:,0]*(1.-x[:,0]))
        u_ex = lambda x: 16.*x[:,0]*(1.-x[:,0])*x[:,1]*(1.-x[:,1])

        # assemble
        fs = Space(mesh, deg, gauss=gauss)
        A = fs.stiffness(cond=True)
        rhs = fs.rhs(f)

        # solution vector
        u = np.zeros(fs.n_dofs())

        bedge_to_dofs = mesh.bedge_to_dofs()
        enforced_dof_nos = np.unique(np.ravel(bedge_to_dofs))

        free_dofs = np.setdiff1d(range(fs.n_dofs()), enforced_dof_nos)   
        n_free_dofs = len(free_dofs)   

        # modify linear system and solve
        A_free = A[free_dofs.reshape(n_free_dofs, 1), free_dofs]
        rhs = rhs - csc_matrix(A).dot(u)

        u[free_dofs] = spla.spsolve(csr_matrix(A_free), rhs[free_dofs])

        # plot solution
        X,Y = mesh.dof_to_coords().T

        if plot:
            fig, (ax1, ax2) = plt.subplots(1,2) #figsize=(10,10),num=j+1
            cont1 = ax1.tricontourf(X, Y, u, 100)            
            plt.colorbar(cont1)

            cont2 = ax2.tricontourf(X, Y, u_ex(mesh.dof_to_coords()), 100)            
            plt.colorbar(cont2)
            plt.show()

    dirichlet_ex(n=32)

    def neuman_ex(n=16, deg=1, gauss=4, diag='right'):
        """ Poisson w/ homogenous Neuman bc in 2D
        ill conditioned -> use CG
        - P1 conv rate: ~ 4
        """
        from scipy.sparse import issparse, csc_matrix, csr_matrix, dia_matrix
        import scipy.sparse.linalg as spla
        import matplotlib.pyplot as plt

        print("\nNeuman ex2 in 2D")
        # data
        mesh = RectangleMesh(nx=n,ny=n,diag=diag)
        pi = math.pi

        f = lambda x: pi*(np.cos(pi*x[:,0]) + np.cos(pi*x[:,1]))
        u_ex = lambda x: (math.cos(pi*x[0]) + math.cos(pi*x[1]))/pi

        # assemble
        fs = Space(mesh, deg, gauss)
        A = fs.stiffness(cond=True)
        rhs = fs.rhs(f)
        print(A)
        # solve
        u, temp = spla.cg(csc_matrix(A), rhs)
        #print(temp); print(u)

        X,Y = mesh.dof_to_coords().T

        fig, ax = plt.subplots() #figsize=(10,10),num=j+1
        cont = ax.tricontourf(X, Y, u, 100)            
        plt.colorbar(cont)

    #neuman_ex()
#--------------------------------------------------------------------------------------#

    def p0_test():
        """ test with P0 element """
        u_ex = lambda x: 16.*x[0]*(1.-x[0])*x[1]*(1.-x[1])      # exact solution

        mesh = UnitSquareMesh(16,16)
        fs = Space(mesh, Element('lagrange', 0), gauss=4)

        A = fs.mass()
        rhs = fs.rhs(u_ex)

        u = fem_solver(fs, A, rhs)
        plot_sol(fs,u,u_ex,contour=False)

#--------------------------------------------------------------------------------------#

    def poisson_test_2d():
        """ Poisson problem 2D test with homogenous Dirichlet bc """
        u_ex = lambda x: 16.*x[0]*(1.-x[0])*x[1]*(1.-x[1])      # exact solution
        f = lambda x: 32.*(x[1]*(1.-x[1]) + x[0]*(1.-x[0]))     # right hand side

        mesh = UnitSquareMesh(16,16,diag='right')
        fs = Space(mesh, Element('lagrange', deg=1), gauss=4)

        A = fs.stiffness()
        rhs = fs.rhs(f)

        bc = Dirichlet(fs,0)
        u = fem_solver(fs, A, rhs, bc)
        plot_sol(fs,u,u_ex,contour=False)


#--------------------------------------------------------------------------------------#

    #p0_test()
    #poisson_test_2d()

