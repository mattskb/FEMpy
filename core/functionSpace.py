import numpy as np
from finiteElement import Element

class Space:
    """Function space class
    Input:
        mesh    - mesh it is built on
        deg 
        gauss   - degree of Gaussian quadrature (optional, default is 4)
    """
    def __init__(self, mesh, deg=None, gauss=4):
        # try to get degree from mesh or set to 1 (piecewise linear)
        if deg is None:
            try:
                deg = mesh.deg()
            except AttributeError:
                deg = 1

        # set data
        self.mesh = mesh  
        self.__deg = deg
        self.__gauss = gauss                
        self.__fe = Element(deg, gauss)       

    def assemble(self, derivative, c=None, cond=False):
        """Assemble linear system
        Input:
            c           - coefficient (variable or constant), can be tensor valued if appropriate
            derivative  - True if derivative of shape function
            cond        - True if condition number is calculated and printed
        Output:
            A           - stiffness/mass matrix (dense)"""

        A = np.zeros((self.mesh.n_dofs(), self.mesh.n_dofs()))            # initialize global assembly matrix
        # loop over elements
        for vc, dc, d in zip(self.mesh.elt_to_vcoords(),\
                             self.mesh.elt_to_dofcoords(),\
                             self.mesh.elt_to_dofs()): 

            self.__fe.set_data(vc, dc)                          # give element data to FE
            elt_assemble = self.__fe.assemble(c, derivative)    # element assembly
            A[np.array([d]).T, d] += elt_assemble               # add to global assembly
        if cond:
            from numpy.linalg import cond
            print("Condition number of {} matrix: {}".format("stiffness" if derivative else "mass", cond(A)))
        return A


    def stiffness(self, c=None, cond=False):
        """Assemble stiffness matrix"""
        return self.assemble(True, c, cond)

    def mass(self, c=None, cond=False):
        """Assemble mass matrix"""
        return self.assemble(False, c, cond)

    def rhs(self, f):
        """Assemble "right hand side" load vector 
        Input:
            f - function (or constant)
        Output:
            load vector
        """
        rhs = np.zeros(self.mesh.n_dofs())
        if f == 0:
            return rhs
        else:
            # loop over elements
            for vc, dc, d in zip(self.mesh.elt_to_vcoords(),\
                                 self.mesh.elt_to_dofcoords(),\
                                 self.mesh.elt_to_dofs()):       
                self.__fe.set_data(vc, dc)                  # give element data to FE
                rhs[d] += self.__fe.assemble_rhs(f)
            return rhs

    def norm(self, u, f=None, p=2):
        """calculate norm of, ||u|| or of ||u - f||
        Input:
            u   - vector of values at nodes"""

        norm = 0
        # loop over elements
        for vc, dc, d in zip(self.mesh.elt_to_vcoords(),\
                             self.mesh.elt_to_dofcoords(),\
                             self.mesh.elt_to_dofs()): 

            self.__fe.set_data(vc, dc)                          # give element data to FE
            norm += self.__fe.norm(u[d], f=f, p=p)
        return np.power(norm,1./p)
        

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
    import math
    from meshing import RectangleMesh

    def dirichlet_ex(data, n=16, deg=1, gauss=4, diag='r', plot=True, cond=False):
        """ Poisson w/ homogenous Dirichlet bc in 2D
        """
        from scipy.sparse import issparse, csc_matrix, csr_matrix, dia_matrix
        import scipy.sparse.linalg as spla
        import matplotlib.pyplot as plt

        #print("\nPoissin eq. w/ Dirichlet bc in 2D")
        # data
        mesh = RectangleMesh(nx=n,ny=n,deg=deg,diag=diag)

        f = data["rhs"]; u_ex = data["u_ex"]

        # assemble
        fs = Space(mesh, deg, gauss=gauss)
        A = fs.stiffness(cond=cond)
        rhs = fs.rhs(f)


        bedge_to_dofs = mesh.bedge_to_dofs()
        enforced_dof_nos = np.unique(np.ravel(bedge_to_dofs))

        free_dofs = np.setdiff1d(range(mesh.n_dofs()), enforced_dof_nos)   
        n_free_dofs = len(free_dofs)   

        # solution vector
        u = np.zeros(mesh.n_dofs())
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

        return fs.norm(u,f=u_ex)

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
        #u, info = spla.cg(csr_matrix(A), rhs)
        #print("CG successful") if info == 0 else print("No convergence")
        u = spla.spsolve(csr_matrix(A), rhs)
        

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

#--------------------------------------------------------------------------------------#

    def conv_test(ex_no=0, deg=1, gauss=4, diag="r"):
        e = []; r = ["-"]; j = 0

        print("Convergence test for Poisson equation with Dirichlet boundary condition:")
        print("\t- deg: ", deg, ", gauss: ", gauss, ", diag: ", diag, "\n")
        print("--------------------")
        template = "{0:<5}{1:<10}{2:<30}"
        print(template.format("res","error","rate"))
        print("--------------------")
        e_prev = 0
        for n in [4,8,16,32,64]:
            e = dirichlet_ex(dirichlet_data(ex_no), n=n, deg=deg, gauss=gauss, diag=diag, plot=False)
            r = e_prev/e
            if r == 0:
                r = "-"
            else:
                r = round(r,2)
            print(template.format(n,round(e,4),r))
            e_prev = e

    
    conv_test(ex_no=4, deg=1, gauss=4, diag="l")


    #neuman_ex(n=16, deg=1, gauss=4, diag='r', plot=True)