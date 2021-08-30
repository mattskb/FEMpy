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

    def assemble(self, derivative, c=None, cond=False):
        """Assemble linear system
        Input:
            c           - coefficient (variable or constant), can be tensor valued if appropriate
            derivative  - True if derivative of shape function
            cond        - True if condition number is calculated and printed
        Output:
            A           - stiffness/mass matrix (dense)
        """
        A = np.zeros((self.n_dofs(), self.n_dofs()))                  # initialize global assembly matrix

        for vc, dc, d in zip(self.__mesh.elt_to_vcoords(),\
                             self.__mesh.elt_to_dofcoords(),\
                             self.__mesh.elt_to_dofs()): 

            self.__fe.set_data(vc, dc)                                # give element data to FE
            elt_assemble = self.__fe.assemble(c, derivative)          # element assembly
            A[np.array([d]).T, d] += elt_assemble                     # add to global assembly
        if cond:
            from numpy.linalg import cond
            print("Condition number of {} matrix: {}".format("stiffness" if derivative else "mass", cond(A)))
        return A


    def stiffness(self, c=None, cond=False):
        """ assemble stiffness matrix """
        return self.assemble(True, c, cond)

    def mass(self, c=None, cond=False):
        """ assemble mass matrix """
        return self.assemble(False, c, cond)

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
            for vc, dc, d in zip(self.__mesh.elt_to_vcoords(),\
                                 self.__mesh.elt_to_dofcoords(),\
                                 self.__mesh.elt_to_dofs()):       
                self.__fe.set_data(vc, dc)      # give element data to FE
                rhs[d] += self.__fe.assemble_rhs(f)
            return rhs

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
    import math
    from meshing import RectangleMesh

    def dirichlet_ex(data, n=16, deg=1, gauss=4, diag='r', plot=True):
        """ Poisson w/ homogenous Dirichlet bc in 2D
        """
        from scipy.sparse import issparse, csc_matrix, csr_matrix, dia_matrix
        import scipy.sparse.linalg as spla
        import matplotlib.pyplot as plt

        print("\nPoissin eq. w/ Dirichlet bc in 2D")
        # data
        mesh = RectangleMesh(nx=n,ny=n,deg=deg,diag=diag)

        f = data["rhs"]; u_ex = data["u_ex"]

        # assemble
        fs = Space(mesh, deg, gauss=gauss)
        A = fs.stiffness(cond=True)
        rhs = fs.rhs(f)


        bedge_to_dofs = mesh.bedge_to_dofs()
        enforced_dof_nos = np.unique(np.ravel(bedge_to_dofs))

        free_dofs = np.setdiff1d(range(fs.n_dofs()), enforced_dof_nos)   
        n_free_dofs = len(free_dofs)   

        # solution vector
        u = np.zeros(fs.n_dofs())
        U_ex = u_ex(mesh.dof_to_coords())
        u[enforced_dof_nos] = U_ex[enforced_dof_nos]

        # modify linear system and solve
        A_free = A[free_dofs.reshape(n_free_dofs, 1), free_dofs]
        rhs = rhs - csc_matrix(A).dot(u)

        u[free_dofs] = spla.spsolve(csc_matrix(A_free), rhs[free_dofs])

        # plot solution
        X,Y = mesh.dof_to_coords().T

        if plot:
            #tri = Triangulation(X, Y, elt_to_vertex)
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.plot_trisurf(X, Y, U_ex, cmap=plt.cm.Spectral)
            ax1.set_title("Exact")
            #ax1.set_xlim(0, 1)
            #ax1.set_ylim(0, 1)
            #ax1.set_zlim(np.min(U_ex), np.max(U_ex))

            ax2 = fig.add_subplot(1, 2, 2, sharez=ax1, sharey=ax1, sharex=ax1, projection='3d')
            ax2.plot_trisurf(X, Y, u, cmap=plt.cm.Spectral) #triangles=tri.triangles, cmap=plt.cm.Spectral)
            ax2.set_title("Numerical")
            #ax2.set_zlim(0, 1)

            plt.show()

#--------------------------------------------------------------------------------------#

    def neuman_ex(n=16, deg=1, gauss=4, diag='r', plot=True):
        """ Poisson w/ homogenous Neuman bc in 2D
        ill conditioned -> use CG
        - P1 conv rate: ~ 4
        """
        from scipy.sparse import issparse, csc_matrix, csr_matrix, dia_matrix
        import scipy.sparse.linalg as spla
        import matplotlib.pyplot as plt

        print("\nNeuman ex2 in 2D")
        # data
        mesh = RectangleMesh(nx=n,ny=n,deg=deg,diag=diag)
        pi = math.pi

        f = lambda x: pi*(np.cos(pi*x[:,0]) + np.cos(pi*x[:,1]))
        u_ex = lambda x: (np.cos(pi*x[:,0]) + np.cos(pi*x[:,1]))/pi

        # assemble
        fs = Space(mesh, deg, gauss)
        A = fs.stiffness(cond=True)
        rhs = fs.rhs(f)
 
        # solve
        u, temp = spla.cg(csr_matrix(A), rhs)
        print(temp)
        #u = spla.spsolve(csr_matrix(A), rhs)
        

        X,Y = mesh.dof_to_coords().T

        if plot:
            #tri = Triangulation(X, Y, elt_to_vertex)
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot_trisurf(X, Y, u, cmap=plt.cm.Spectral) #triangles=tri.triangles, cmap=plt.cm.Spectral)
            ax.set_title("Numerical")
            #ax.set_zlim(0, 1)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.plot_trisurf(X, Y, u_ex(mesh.dof_to_coords()), cmap=plt.cm.Spectral)
            ax.set_title("Exact")
            #ax.set_zlim(0, 1)
            

            plt.show()
    
#--------------------------------------------------------------------------------------#

    def gauss_test(n=16, deg=1, gauss=4, diag='r'):
        # data
        mesh = RectangleMesh(nx=n,ny=n,deg=deg,diag=diag)

        f = lambda x: 32.*(x[:,0]*(1.-x[:,0]) + x[:,1]*(1.-x[:,1]))
        u_ex = lambda x: 16.*x[:,0]*(1.-x[:,0])*x[:,1]*(1.-x[:,1])
        f = lambda x: 4

        # assemble
        fs_g1 = Space(mesh, deg, gauss=gauss, gcheck=True)
        A_g1 = fs_g1.stiffness()
        M_g1 = fs_g1.mass()
        rhs_g1 = fs_g1.rhs(f)

        fs = Space(mesh, deg, gauss=gauss, gcheck=False)
        A = fs.stiffness()
        M = fs.mass()
        rhs = fs.rhs(f)

        print("stiffness check: ", np.allclose(A,A_g1))
        print("mass check: ", np.allclose(M,M_g1))
        print("rhs check: ", np.allclose(rhs,rhs_g1))
        print(A)
        print(A_g1)
        print(rhs)
        print(rhs_g1)
        #mesh.plot()


    def dirichlet_data(n):
        from math import pi
        rhs = [lambda x: 32.*(x[:,0]*(1.-x[:,0]) + x[:,1]*(1.-x[:,1])),
               -6,
               lambda x: 2*pi*pi*np.sin(pi*x[:,0])*np.cos(pi*x[:,1]),
               lambda x: -2*(x[:,0]**2 - x[:,0] + x[:,1]**2 - x[:,1] + 0.5),
               lambda x: -6*(x[:,0] + x[:,1])]

        u_ex = [lambda x: 16.*x[:,0]*(1.-x[:,0])*x[:,1]*(1.-x[:,1]),
                lambda x: 1 + np.power(x[:,0],2) + 2*np.power(x[:,1],2),
                lambda x: np.sin(pi*x[:,0])*np.cos(pi*x[:,1]), 
                lambda x: np.power((x[:,0]-0.5),2) * np.power((x[:,1]-0.5),2),
                lambda x: np.power(x[:,0],3) + np.power(x[:,1],3)]

        return {"rhs": rhs[n], "u_ex": u_ex[n]}


    dirichlet_ex(dirichlet_data(3), n=16, deg=1, gauss=4, diag='r', plot=True)

    #neuman_ex(n=16, deg=1, gauss=4, diag='r', plot=True)
    #gauss_test(n=2, deg=1, gauss=4, diag='l')

    


