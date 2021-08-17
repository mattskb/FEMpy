import numpy as np
from numpy.linalg import norm
from gaussquad import gaussPts
from shapefuncs import ShapeFunctions

#--------------------------------------------------------------------------------------#

def affine_map(vertices):
    """ affine transformation F : K_ref -> K, where F(x) = a x + b, for x in K_ref
    Input:
        vertices - local vertices to coordinate matrix
    Output:
        B, b, det (2D: meas = |det|/2, 1D: meas = |det|)
    """
    if len(vertices) == 3:
        tmp1 = vertices[1] - vertices[0]
        tmp2 = vertices[2] - vertices[0]

        a = np.array([tmp1, tmp2]).T
        a_inv = np.linalg.inv(a)
        det = np.linalg.det(a)
        b = vertices[0]
    else:
        tmp1 = (vertices[1] - vertices[0])/2
        tmp2 = 2*np.ones(2)

        a = np.array([tmp1, tmp2]).T
        #a_inv = np.linalg.inv(a)
        a_inv = None
        det = np.linalg.det(a)
        b = (vertices[0] + vertices[1])/2
    return a, a_inv, det, b

#--------------------------------------------------------------------------------------#

class Element:
    """ Finite Element class
    Input:
            deg - degree of piecewise polynomial (int)
            gauss - degree of Gaussian quadrature
    """
    def __init__(self, deg, gauss):


        self.__deg = deg
        self.__gauss = gauss
        self.__sfns = ShapeFunctions(self.__deg)

    def initialized(self):
        """ return True if element is initialized """
        return self.__initialized


    def deg(self):
        """ return degree of element """
        return self.__deg


    def initialize(self, gauss):
        """ initialize finite element
        Input:
            dim - spatial dimension of finite element
            gauss - degree of Gaussian quadrature
        """
        sfns = ShapeFunctions(self.__deg); n_dofs = sf.n_dofs()

        self.__gauss = gauss
        self.__sfns = sfns
        self.__n_dofs = n_dofs
        self.__initialized = True

    def n_dofs(self):
        """ return number of dofs """
        return self.__sfns.n_dofs()

    def set_data(self, vertices, signs=None):
        """ set local vertex to coordinate matrix """

        # initialize affine transform
        a, a_inv, det, b = affine_map(vertices)
        self.__a = a
        self.__a_inv = a_inv
        self.__det = det
        self.__b = b

        # set data
        self.__vertices = vertices
        self.__signs = signs

    def measure(self):
        """ returns measure of element (area in 2d, length in 1d) """
        return abs(self.__det)/self.__dim

    def map_to_elt(self, xi):
        """ maps ref coord xi to global coord x """
        return np.dot(self.__a, xi) + self.__b

    def map_to_ref(self, x):
        """ maps global coord x to local coord xi of reference element """
        return np.linalg.solve(self.__a, x - self.__b)

    def integrate(self, f, gauss=None):
        """ integrate (global) function f over element
        Input:
            f - function to be integrated (callable or constant)
            gauss - gauss - degree of Gaussian quadrature (default: same as element)
        Output:
            integral
        """
        if gauss is None:
            gauss = self.__gauss

        if callable(f):
            # get quadrature points and weights
            xw = np.array(gaussPts(gauss, self.__dim))

            nq = xw.shape[0]             # number of quad points
            qweights = xw[:,2]           # quad weights
            qpoints = xw[:,:2]           # quad points

            # calculate and return integrals
            return sum([ f( self.map_to_elt(qpoints[n]) )*qweights[n] for n in xrange(nq) ])*self.measure()*(self.__dim/2.)
        else:
            return f*self.measure()

    def jacobi(self):
        """ return Jacobi matrix of transformation x -> xi, i.e. dxi/dx """
        if self.__dim == 2:
            return self.__a_inv
        else:
            return 2./self.measure()

    def eval(self, n, x):
        """ evaluate shape function at global coord x
        Input:
            n - n'th shape function to be evaluated
            x - global coord
        """
        if self.__deg == 0:
            return 1.
        else: # lagrange of order 1,2 or 3
            xi = self.map_to_ref(x)
            return self.__sfns[k].eval(n, xi, derivative=False)

    def deval(self, n, x):
        """ evaluate shape function derivative at global coord x
        Input:
            n - n'th shape function to be evaluated
            x - global coord
        """
        if self.__deg == 0:
            raise ValueError('No derivative of P0 shape functions.')
        elif self.__deg == 1:
            v1 = self.__vertices[(n+1)%3]; v2 = self.__vertices[(n+2)%3]
            return np.array([ v1[1] - v2[1], v2[0] - v1[0] ])/abs(self.__det)
        else: # lagrange of order 2 or 3
            xi = self.map_to_ref(x)
            return np.dot(self.__sfns[k].eval(n, xi, derivative=True), self.jacobi())

    def evaluate(self, n, x, derivative=False, m=0):
        """ evaluate shape functions """
        if derivative:
            return self.deval(n, x, m)
        else:
            return self.eval(n, x, m)

    def assemble_stress(self, d1=0, d2=0, c=None):
        """ Assemble parts of stress tensor (for element), i.e. (partial_x phi_i, partial_y phi_j)
        *only call this for P1 elements in 2D*
        Input:
            d1 - partial deriv for row index (0 for x, 1 for y)
            d2 - partial deriv for column index (0 for x, 1 for y)
        """
        v0, v1, v2 = self.__vertices

        # rows are gradients of phi
        phis = np.array([[ v1[1] - v2[1], v2[0] - v1[0] ],\
                         [ v2[1] - v0[1], v0[0] - v2[0] ],\
                         [ v0[1] - v1[1], v1[0] - v0[0] ]])

        A = np.tensordot(phis[:,d1], phis[:,d2], 0)
        if c is None:
            return A/(4*self.measure())
        else:
            return A*self.integrate(c)/(self.__det**2)
                
    def assemble(self, c=None, derivative=[False,False], m=0, n=0):
        """
        integrate shape function products over this element
        Input:
            c - coefficient (variable or constant) can be tensor valued if 2D
            derivative - True if derivative of shape function evaluated
            m, n - methods
        Output:
            A - matrix of integrals
        """
        if self.__m_keys[m] == (0,'rt') and self.__m_keys[n] == (0,'lagrange') and derivative == [True,False]:
            return self.assemble_dRT0_P0(c)
        elif self.__m_keys[m] == (0,'lagrange') and self.__m_keys[n] == (0,'rt') and derivative == [False,True]:
            return self.assemble_dRT0_P0(c).T
        elif m == n and self.__m_keys[m] == (0,'lagrange') and derivative == [False,False]:
            return self.assemble_P0_P0(c)
        elif m == n and self.__m_keys[m] == (1,'lagrange') and derivative == [True,True]:
            return self.assemble_dP1_dP1(c)
        else:
            A = np.zeros((self.n_dofs(m), self.n_dofs(n)))
            for i in xrange(self.n_dofs(m)):
                for j in xrange(self.n_dofs(n)):
                    if c is None:
                        A[i,j] = self.integrate(lambda x: np.dot(self.evaluate(i,x,derivative[0],m),\
                                                                 self.evaluate(j,x,derivative[1],n)))
                    elif not callable(c):
                        A[i,j] = self.integrate(lambda x: np.dot(np.dot(c, self.evaluate(i,x,derivative[0],m)),\
                                                                           self.evaluate(j,x,derivative[1],n)))
                    else: # callable c
                        A[i,j] = self.integrate(lambda x: np.dot(np.dot(c(x), self.evaluate(i,x,derivative[0],m)),\
                                                                              self.evaluate(j,x,derivative[1],n)))
            return A

    def rhs(self, f, m=0):
        """ assemble element right hand side """
        if self.__m_keys[m] == (0,'lagrange'):
            if callable(f):
                return np.array([self.integrate(f)])
            else: # not callable
                return np.array([f*self.measure()])
        else:
            if callable(f):
                return np.array([self.integrate( lambda x: np.dot(f(x), self.eval(n,x,m)) ) for n in xrange(self.n_dofs(m))])
            else:
                return np.array([self.integrate(lambda x: self.eval(n,x,m))*f for n in xrange(self.n_dofs(m))])


#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
    from meshing import UnitSquareMesh
    fe = Element()
    fe.initialize(2, 4)
    #vs = np.array([[0,0], [1,0], [0,1]])
    #fe.set_data(vs)
    mesh = UnitSquareMesh(8,8)

    for j in xrange(mesh.n_elts()):
        vs = mesh.elt_to_vcoords(j)
        fe.set_data(vs)
        print "\n--------- element {} ------------".format(j)
        print "measure: ", fe.measure()
        print "det: ", fe._Element__det
        for v in vs:
            print fe.true_deval(0,v), fe.deval(0,v), fe.eval(0,v)
            print fe.true_deval(1,v), fe.deval(1,v), fe.eval(1,v)
            print fe.true_deval(2,v), fe.deval(2,v), fe.eval(2,v), "\n"

            # assert np.allclose(fe.true_deval(0,v) - fe.deval(0,v),0)
            # assert np.allclose(fe.true_deval(1,v) - fe.deval(1,v),0)
            # assert np.allclose(fe.true_deval(2,v) - fe.deval(2,v),0)

        stress = fe.assemble_stress()




