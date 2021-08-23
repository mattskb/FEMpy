import numpy as np
from scipy.sparse import issparse, csc_matrix, csr_matrix, dia_matrix
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from boundaryCond import Dirichlet, Neuman
"""
------------------------------------------------------------------------------------
Some auxiliary useful functions
    fem_solver - solves linear system with Dirichlet or Neuman (enforced) boundary conditions
    plot_sol - plots solution, 1d or 2d
    plot_piecewise - makes piecewise plot of 1d solution (for P0 elts)
------------------------------------------------------------------------------------
"""
def get_solver(name):
    """ returns the correct solver passed as string arg to fem solver """
    name = name.lower()

    if name == 'cg':
        return spla.cg
    elif name == 'gmres':
        return spla.gmres
    elif name =='minres':
        return spla.minres
    else:
        raise ValueError('Invalid solver argument.')

#--------------------------------------------------------------------------------------#

def inc_dirichlet(bcs, n_dofs):
    """ help incorporate enforced Dirichlet bc
    Input:
        bcs - Dirichlet boundary conditions
        n_dofs - total number of DOFs in function space
    Output:
        u - initialized solution vector
        dirichlet_nodes - unique node numbers of nodes where bc is enforced
    """
    # initialize solution vector
    u = np.zeros(n_dofs)

    # put unique Dirichlet nodes here
    dirichlet_nodes = np.array([], dtype=int)

    # inc bc in solution vector and save nodes
    for bc in bcs:
        unique_nodes = bc.unique_dof_nos()                              # unique node numbers in this bc
        new_nodes = np.setdiff1d(unique_nodes, dirichlet_nodes)         # new nodes not in previous bc
        u[new_nodes] += bc.assemble()[new_nodes]                        # update solution vector
        dirichlet_nodes = np.concatenate((dirichlet_nodes, new_nodes))

    return u, dirichlet_nodes

#--------------------------------------------------------------------------------------#

def inc_neuman(bcs, n_dofs):
    """ help incorporate enforced Neuman bc
    Input:
        bcs - Neuman boundary conditions
        n_dofs - total number of DOFs in function space
    Output:
        u - initialized solution vector
        neuman_edges - unique edge numbers of edges where bc is enforced
    """

    # initialize solution vector
    u = np.zeros(n_dofs)

    # put unique Neuman edges here
    neuman_edges = np.array([], dtype=int)

    for bc in bcs:
        edges = bc.unique_dof_nos()
        u += bc.assemble()
        neuman_edges = np.concatenate((neuman_edges, edges))

    return u, neuman_edges

#--------------------------------------------------------------------------------------#

def fem_solver(fs, A, rhs, *bcs, **kw):
    """
    Solve linear system Au = rhs with specified Dirichlet or Neuman boundary conditions (optional)
    Input:
        fs - function space
        A - linear system (sparse if not mixed, dense if mixed)
        rhs - right hand side vector
        bcs - Dirichlet (for lagrange elt) or Neuman (for RT elt) boundary conditions (optional)
        kw:
            info - print some information about linear system (condition number, size and rank)
            solver - name of linear solver to be used (default: spsolve)
            tol - tolerance of iterative method (default: 1e-06)
            P - preconditioner for iterative method (default: none)
    Output:
        u - solution vector, if not mixed
        (m_1,u_1),...,(m_n,u_n) - if mixed (i.e. tuples of FE number and solution vector)
    """
    # get kw
    info = kw.pop('info',False); solver_name = kw.pop('solver', None); tol = kw.pop('tol', 1e-06); # P = kw.pop('P',None)
    x0 = kw.pop('x0', None)

    if len(bcs) != 0: # <------------------------------- incorporate boundary conditions (if any)

        # separate Dirichlet and Neuman boundary conditions
        neuman = []; dirichlet = []
        for bc in bcs:
            if isinstance(bc,Dirichlet):
                dirichlet.append(bc)
            else:
                neuman.append(bc)

        # assemble Dirichlet (for Lagrange element)
        if dirichlet == []:
            uD = np.zeros(fs.n_dofs()); dirichlet_node_nos = np.array([],dtype=int)
        else:
            # get updated solution vector and unique dof numbers
            uD, dirichlet_node_nos = inc_dirichlet(dirichlet, fs.n_dofs())

        # assemble Neuman (for RT element)
        if neuman == []:
            uN = np.zeros(fs.n_dofs());  neuman_edge_nos = np.array([],dtype=int)
        else:
            # get updated solution vector and unique dof numbers
            uN, neuman_edge_nos = inc_neuman(neuman, fs.n_dofs())

        u = uD + uN
        enforced_dof_nos = np.concatenate((dirichlet_node_nos, neuman_edge_nos))   # unique dof numbers of enforced dofs
        free_dofs = np.setdiff1d(range(fs.n_dofs()), enforced_dof_nos)             # unique dof numbers of free dofs
        n_free_dofs = len(free_dofs)                                               # number of free nodes

        # convert linear system to dense (if needed)
        if issparse(A):
            A = A.todense()

        # modify linear system
        A_free = A[free_dofs.reshape(n_free_dofs, 1), free_dofs]
        rhs = rhs - csc_matrix(A).dot(u)

        # # set up preconditioner (if needed)
        # if P is not None:
        #     M_x = lambda x: spla.spsolve(csc_matrix(P[free_dofs.reshape(n_free_dofs, 1), free_dofs]), x)
        #     M = spla.LinearOperator((n_free_dofs, n_free_dofs), M_x)
        # else:
        #     M = None

        # print condition number
        if info:
            print "Condition number of matrix: ", np.linalg.cond(A_free)
            print "Rank of matrix: ", np.linalg.matrix_rank(A_free)
            print "Shape of matrix: ", A_free.shape

        # solve
        if solver_name is None:
            u[free_dofs] = spla.spsolve(csr_matrix(A_free), rhs[free_dofs])
        else:
            solver = get_solver(solver_name)
            if x0 is not None:
                x0 = x0[free_dofs]
            u[free_dofs], temp = solver(csc_matrix(A_free), rhs[free_dofs], x0=x0, tol=tol, maxiter=A_free.shape[0])
            print temp
            assert temp == 0, 'Iterative solver, {}, failed to converge.'.format(solver_name)

    else: # <--------------------------------------------- no boundary conditions
        # print condition number
        if info:
            print "Condition number of matrix: ", np.linalg.cond(A)
            print "Rank of matrix: ", np.linalg.matrix_rank(A_free)
            print "Shape of matrix: ", A_free.shape

        # # set up preconditioner (if needed)
        # if P is not None:
        #     M_x = lambda x: spla.spsolve(csc_matrix(P), x)
        #     M = spla.LinearOperator(P.shape, M_x)
        # else:
        #     M = None

        # solve
        if solver_name is None:
            u = spla.spsolve(csc_matrix(A), rhs)
        else:
            solver = get_solver(solver_name)
            # u, temp = solver(csc_matrix(A), rhs, M=M, tol=tol)
            u, temp = solver(csc_matrix(A), rhs, x0=x0, tol=tol)
            assert temp == 0, 'Iterative solver, {}, failed to converge.'.format(solver_name)

    # return solution vector(s)
    if fs.mixed():
        us = []; n0 = 0; n = 0
        for j in xrange(fs.n_FE()):
            n += fs.n_dofs(j)
            us.append((j,u[n0:n]))
            n0 = n
        return us
    else:
        return u

#--------------------------------------------------------------------------------------#

def plot_sol(fs, u, u_ex=None, **kw):
    """ makes a plot of u on the mesh the function space is built on
    Input:
        fs - function space
        u - solution vector (values to be plotted)
        u_ex - exact solution (callable)
        kw:
            contour - True if contour plot, else 3D plot (default: True)
            name - name of figure (default: Plotting)
    """
    # get kwargs
    contour = kw.pop('contour', True); name = kw.pop('name', 'Plotting')

    # get solution vector
    if isinstance(u, tuple):
        assert fs.mixed(), 'Wrong format of solution vector, space is not mixed.'
        m = u[0]; u = u[1]
    else:
        m = 0

    # set up figure
    if u_ex is None:
        fig = plt.figure(num=name)
    else:
        fig = plt.figure(figsize=(12,6), num=name)

    # get plotting points from function space
    coords = fs.dof_to_cn(m)
    X = coords[:,0]; Y = coords[:,1]

    # plot solution
    if fs.mesh.dim() == 2:
        if u_ex is None:
            if not contour: # 3D plot
                ax = fig.gca(projection='3d')
                ax.plot_trisurf(X, Y, u, linewidth=0.2, antialiased=True, cmap=plt.cm.Spectral)
            else: # contour plot
                plot = plt.tricontourf(X,Y,u,100)
                CB = plt.colorbar(plot, shrink=0.8, extend='both')

            plt.title("computed solution")
            plt.xlabel('x'); plt.ylabel('y')
            plt.xticks([]); plt.yticks([])
        else:
            U_ex = [u_ex(c) for c in coords]
            if not contour: # 3D plot
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                ax.plot_trisurf(X, Y, u, linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
                plt.title("computed solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')

                ax = fig.add_subplot(1, 2, 2, projection='3d')
                ax.plot_trisurf(X, Y, U_ex, linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
                plt.title("exact solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')
            else: # contour plot
                plt.subplot(121)
                plot = plt.tricontourf(X,Y,u,100)
                CB = plt.colorbar(plot, shrink=0.8, extend='both')
                plt.title("computed solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')

                plt.subplot(122)
                plot = plt.tricontourf(X,Y,U_ex,100)
                CB = plt.colorbar(plot, shrink=0.8, extend='both')
                plt.title("exact solution")
                plt.xticks([]); plt.yticks([])
                plt.xlabel('x'); plt.ylabel('y')

    else: # dim = 1
        if u_ex is None:
            plt.plot(X, u)
            plt.title("computed solution")
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            plt.subplot(121)
            plt.plot(X, u)
            plt.title("computed solution")
            plt.xlabel('x')
            plt.ylabel('y')

            U_ex = [u_ex(c) for c in coords]

            plt.subplot(122)
            plt.plot(X, U_ex)
            plt.title("exact solution")
            plt.xlabel('x')
            plt.ylabel('y')

    plt.show()

#--------------------------------------------------------------------------------------#

def plot_piecewise(u, nodes, u_ex=None, name=''):
    """ makes a piecewise plot of 1D solution
    Input:
        u - computed solution
        mesh - 1d mesh
        u_ex - exact solution (optional)
        name - name of figure """
    assert np.allclose(nodes[:,1],0), 'Piecewise plotting only for 1 dim.'
    nodes = nodes[:,0]

    # set up figure
    if name == '':
        name = 'Plot piecewise constant in 1D'
    fig = plt.figure(num=name)

    x = np.linspace(0,1,101)
    vertices = np.unique(mesh.vertices()) # <------- fix this line
    n_v = len(vertices)
    plt.plot(x, np.piecewise(x, [(x >= vertices[n]) & (x <= vertices[n+1]) for n in xrange(n_v-1)], u))

    # evaluate exact solution at nodes
    if u_ex is not None:
        U_ex = []
        for p in x:
            U_ex.append(u_ex([p]))

        plt.plot(x, U_ex)
        plt.legend(('computed','exact'))
    else:
        plt.legend('computed')

    plt.show()

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':

    def piecewise_test(n=10):
        """ test piecewise plotting """
        from meshing import UnitIntMesh
        from functionSpace import FunctionSpace
        from finiteElement import FiniteElement

        u_ex = lambda x: 4*x[0]*(1.-x[0])                           # exact solution
        mesh = UnitIntMesh(n)                                       # 1d mesh
        fs = FunctionSpace(mesh, FiniteElement('lagrange', 0))      # P0 function space

        # assemble
        A = fs.mass()
        rhs = fs.rhs(u_ex)

        # solve with homogenous Dirichlet bc
        u = fem_solver(fs, A, rhs)
        plot_piecewise(u, fs.dof_to_coords(), u_ex)

    #piecewise_test()


