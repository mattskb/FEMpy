import numpy as np

class Dirichlet:
    """ class for implementing enforced Dirichlet boundary conditions for Lagrange element
    Input:
        fs - function space
        g - funciton (or constant) on the boundary
        names - names for boundary dictionary (default: entire boundary)
        kw:
            m - number of FE in case of mixed space (default: 0)
    """
    def __init__(self, fs, g, *names, **kw):
        """ assembles Dirichlet boundary condition for Lagrange element,
        given by function (or constant) g on the boundary.
        parts of boundary can be specified according to names in boundary dictionary """
        m = kw.pop('m', 0)
        assert fs.fe.method(m) == 'lagrange', 'This class is only for implementing Dirichlet BC for Lagrange element.'

        # get data from function space
        dnode_nos_m = fs.ext_dof_nos(*names, m=m)                           # local dirichlet node numbers
        node_to_coords = fs.dof_to_cn(m)                                    # node number to coords matrix

        # assemble boundary condition
        bc = np.zeros(fs.n_dofs())                                          # initialize boundary condition
        dnode_nos = dnode_nos_m + sum([fs.n_dofs(j) for j in xrange(m)])    # global dirichlet node numbers
        if callable(g):
            bc[dnode_nos] = [g(node_to_coords[n]) for n in dnode_nos_m]
        else:
            bc[dnode_nos] = g

        self.__unique_dof_nos = dnode_nos
        self.__bc = bc
        self.__m = m

    def m(self):
        """ return method number """
        return self.__m

    def unique_dof_nos(self):
        """ return list of all node number involved in this bc """
        return self.__unique_dof_nos

    def assemble(self):
        """ assemble boundary condition """
        return self.__bc

class Neuman:
    """ class for implementing Neuman boundary conditions for Raviart-Thomas element
    Input:
        fs - function space
        g - funciton (or constant) on the boundary
        names - names for boundary dictionary (default: entire boundary)
        kw:
            m - number of FE in case of mixed space (default: 0)
    """
    def __init__(self, fs, g, *names, **kw):
        """ assembles Neuman boundary conditions for Raviart-Thomas element,
        given by function (or constant) g on the boundary.
        parts of boundary can be specified according to names in boundary dictionary """
        m = kw.pop('m', 0)
        assert fs.fe.method(m) == 'rt', 'This class is only for implementing Neuman BC for RT element.'

        # assemble boundary condition
        if g == 0:
            bc = np.zeros(fs.n_dofs())
        else:
            bc = fs.assemble_boundary(g, *names, m=m, dirichlet=False)

        # get unique edge nos
        neuman_dof_nos = fs.ext_dof_nos(*names, m=m) + sum([fs.n_dofs(j) for j in xrange(m)])

        self.__m = m
        self.__bc = bc
        self.__unique_dof_nos = neuman_dof_nos

    def m(self):
        """ return method number """
        return self.__m

    def unique_dof_nos(self):
        """ return list of all node number involved in this bc """
        return self.__unique_dof_nos

    def assemble(self):
        """ assemble boundary condition """
        return self.__bc
