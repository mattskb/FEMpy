import numpy as np
from numpy.linalg import norm
from gaussquad import gaussPts
from shapefuncs import ShapeFunctions

def get_args(*args):
    """ get args input for Element class """
    method = []; deg = []

    if len(args) == 0:
        # default is Lagrange of degree 1
        deg.append(1); method.append('lagrange')
    elif all(map(lambda x: isinstance(x,list), args)):
        # if args is as lists of methods and degrees
        if all(map(lambda x: isinstance(x,int), args[0])):
            deg = args[0]; method = args[1]
        else:
            deg = args[1]; method = args[0]
    elif all(map(lambda x: isinstance(x,tuple), args)):
        # if args is as method-degree tuples
        for a in args:
            if isinstance(a[0],int):
                deg.append(a[0]); method.append(a[1])
            else:
                deg.append(a[1]); method.append(a[0])
    else:
        # if just method and tuple
        if isinstance(args[0],int):
            deg.append(args[0]); method.append(args[1])
        else:
            deg.append(args[1]); method.append(args[0])

    # check if valid number of degrees and methods and return
    assert len(method) == len(deg), 'Invalid input.'
    return method, deg

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
            method - finite element method (string)
            deg - degree of finite element method (int)
            or (deg, method) - tuples
    """
    def __init__(self, *args):

        # get args input
        method, deg = get_args(*args)

        # check if mixed elt
        if len(method) > 1:
            self.__mixed = True
        else:
            self.__mixed = False

        # check if valid methods
        method = map(lambda x: x.lower(), method)
        for m in method:
            assert any(map(lambda x: x == m, ['lagrange', 'rt'])), 'Invalid method: {}.'.format(m)

        # set data
        m_keys = []
        for j in xrange(len(method)):
            m_keys.append((deg[j], method[j]))
        self.__m_keys = m_keys

        self.__initialized = False

    def initialized(self):
        """ return True if element is initialized """
        return self.__initialized

    def mixed(self):
        """ True if mixed finite element """
        return self.__mixed

    def m_keys(self, m=None):
        """ return all or m'th degree/method tuples, e.g. (1,'lagrange') """
        if m is None:
            return self.__m_keys
        else:
            return self.__m_keys[m]

    def method(self, m=None):
        """ returns finite element method """
        if m is None:
            return [k[1] for k in self.m_keys()]
        else:
            return self.m_keys(m)[1]

    def deg(self, m=None):
        """ returns degree """
        if m is None:
            return [k[0] for k in self.m_keys()]
        else:
            return self.m_keys(m)[0]

    def initialize(self, dim, gauss):
        """ initialize finite element
        Input:
            dim - spatial dimension of finite element
            gauss - degree of Gaussian quadrature
        """
        sfns = {}; n_dofs = {}
        for k in self.__m_keys:
            if k not in sfns.keys():
                if k == (0,'lagrange'):
                    sf = None; nd = 1
                elif k == (0,'rt'):
                    sf = None; nd = 3
                else:
                    sf = ShapeFunctions(k[0], dim); nd = sf.n_dofs()
                sfns[k] = sf; n_dofs[k] = nd

        self.__dim = dim
        self.__gauss = gauss
        self.__sfns = sfns
        self.__n_dofs = n_dofs
        self.__initialized = True

    def n_dofs(self, m=0):
        """ return number of dofs """
        return self.__n_dofs[self.m_keys(m)]

    def set_data(self, vertices, signs=None):
        """ set local vertex to coordinate matrix (and local edge to sign matrix if RT element) """
        assert len(vertices)-1 == self.__dim, 'Number of vertices and dimension mismatch.'

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

    def integrate(self, f, **kw):
        """ integrate (global) function f over element
        Input:
            f - function to be integrated (callable or constant)
            kw:
                gauss - degree of Gaussian quadrature (default: same as element)
        Output:
            integral
        """

        if callable(f):
            # get quadrature points and weights
            xw = np.array(gaussPts(kw.pop('gauss',self.__gauss), self.__dim))

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

    def eval(self, n, x, m=0):
        """ evaluate shape function at global coord x
        Input:
            n - n'th shape function to be evaluated
            x - global coord
            m - m'th FE
        """
        k = self.__m_keys[m]
        if k == (0,'lagrange'):
            return 1.
        elif k == (0,'rt'):
            e = norm(self.__vertices[(n+1)%3] - self.__vertices[(n+2)%3]) # measure of edge
            return self.__signs[n]*e*(x - self.__vertices[n])/(2*self.measure())
        else: # lagrange of order 1,2 or 3
            xi = self.map_to_ref(x)
            return self.__sfns[k].eval(n, xi, derivative=False)

    def deval(self, n, x, m=0):
        """ evaluate shape function derivative at global coord x
        Input:
            n - n'th shape function to be evaluated
            x - global coord
            m - m'th method
        """
        k = self.__m_keys[m]
        if k == (0,'lagrange'):
            raise ValueError('No derivative of P0 shape functions.')
        elif k == (1,'lagrange'):
            if self.__dim == 2:
                v1 = self.__vertices[(n+1)%3]; v2 = self.__vertices[(n+2)%3]
                return np.array([ v1[1] - v2[1], v2[0] - v1[0] ])/abs(self.__det)
            else: # dim = 1
                return pow(-1,n+1)*1/self.measure()
        elif k == (0,'rt'):
            e = norm(self.__vertices[(n+1)%3] - self.__vertices[(n+2)%3]) # measure of edge
            return self.__signs[n]*e/self.measure()
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

    def assemble_P0_div(self, c=None):
        """ Assemble mixed piecewise const and div of vector P1, i.e. (q, dx u1) or (q, dy u2)
        *only call this for P1 elements in 2D*
        Input:
            d - partial deriv (0 for dx, 1 for dy)
            c - coefficient
        """
        v0, v1, v2 = self.__vertices

        # rows are gradients of phi
        phis = np.array([[ v1[1] - v2[1], v2[0] - v1[0] ],\
                         [ v2[1] - v0[1], v0[0] - v2[0] ],\
                         [ v0[1] - v1[1], v1[0] - v0[0] ]])

        if c is None:
            return phis[:,0]/2, phis[:,1]/2
        else:
            return phis[:,0]*self.integrate(c)/abs(self.__det), phis[:,1]*self.integrate(c)/abs(self.__det)

    def assemble_P0_P0(self, c=None):
        """ efficient assembly of element mass matrix for P0 element """
        if c is None:
            return np.array([[self.measure()]])
        elif not callable(c):
            return c*np.array([[self.measure()]])
        else: # callable c
            return np.array([[self.integrate(c)]])

    def assemble_dP1_dP1(self, c=None):
        """ efficient assembly of element stiffness matrix for P1 element """
        if self.__dim == 1:
            temp = np.array([[1., -1.],\
                             [-1., 1.]])/self.measure()
            if c is None:
                return temp
            elif not callable(c):
                return c*temp
            else: # callable c
                return self.integrate(c)*temp/self.measure()
        else: # dim = 2
            if c is None:
                E = self.__vertices[[1, 2, 0],:] - self.__vertices[[2, 0, 1],:]
                return np.dot(E,E.T)/(4*self.measure())
            elif not callable(c): # possibly tensor coeff
                v0, v1, v2 = self.__vertices

                phi1 = np.array([ v1[1] - v2[1], v2[0] - v1[0] ])
                phi2 = np.array([ v2[1] - v0[1], v0[0] - v2[0] ])
                phi3 = np.array([ v0[1] - v1[1], v1[0] - v0[0] ])

                return np.array([[np.dot(c,phi1).dot(phi1), np.dot(c,phi1).dot(phi2), np.dot(c,phi1).dot(phi3)],\
                                 [np.dot(c,phi2).dot(phi1), np.dot(c,phi2).dot(phi2), np.dot(c,phi2).dot(phi3)],\
                                 [np.dot(c,phi3).dot(phi1), np.dot(c,phi3).dot(phi2), np.dot(c,phi3).dot(phi3)]])/(4*self.measure())
            else: # callable c, possibly tensor valued
                v0, v1, v2 = self.__vertices

                phi1 = np.array([ v1[1] - v2[1], v2[0] - v1[0] ])/abs(self.__det)
                phi2 = np.array([ v2[1] - v0[1], v0[0] - v2[0] ])/abs(self.__det)
                phi3 = np.array([ v0[1] - v1[1], v1[0] - v0[0] ])/abs(self.__det)

                a11 = self.integrate(lambda x: np.dot(c(x), phi1).dot(phi1))
                a12 = self.integrate(lambda x: np.dot(c(x), phi1).dot(phi2))
                a13 = self.integrate(lambda x: np.dot(c(x), phi1).dot(phi3))

                a21 = self.integrate(lambda x: np.dot(c(x), phi2).dot(phi1))
                a22 = self.integrate(lambda x: np.dot(c(x), phi2).dot(phi2))
                a23 = self.integrate(lambda x: np.dot(c(x), phi2).dot(phi3))

                a31 = self.integrate(lambda x: np.dot(c(x), phi3).dot(phi1))
                a32 = self.integrate(lambda x: np.dot(c(x), phi3).dot(phi2))
                a33 = self.integrate(lambda x: np.dot(c(x), phi3).dot(phi3))

                return np.array([[a11, a12, a13],\
                                 [a21, a22, a23],\
                                 [a31, a32, a33]])

    def assemble_dRT0_P0(self, c=None):
        """ efficient assembly of element RT0-P0 matrix """
        e1 = norm(self.__vertices[1] - self.__vertices[2])*self.__signs[0]
        e2 = norm(self.__vertices[2] - self.__vertices[0])*self.__signs[1]
        e3 = norm(self.__vertices[0] - self.__vertices[1])*self.__signs[2]
        if c is None:
            return np.array([[e1,e2,e3]]).T
        elif not callable(c):
            return c*np.array([[e1,e2,e3]]).T
        else: # callable c
            return self.integrate(c)*np.array([[e1,e2,e3]]).T/self.measure()

    def assemble_flux_product(self, w, c=None):
        """ assemble product of fluxes """
        assert self.__m_keys[0] == (0,'rt') and self.__m_keys[1] == (0,'lagrange'), 'Must be mixed space.'
        A = np.zeros(3)
        if c is None:
            for j in xrange(3):
                if not callable(w):
                    f = lambda x: np.dot(w[0]*self.eval(0,x), self.eval(j,x)) \
                                + np.dot(w[1]*self.eval(1,x), self.eval(j,x)) \
                                + np.dot(w[2]*self.eval(2,x), self.eval(j,x))
                    A[j] = self.integrate(f)
                else: # callable w
                    A[j] = self.integrate(lambda x: np.dot(w(x), self.eval(j,x)))
        elif callable(c):
            for j in xrange(3):
                f = lambda x: np.dot(w[0]*self.eval(0,x), np.dot(c(x),self.eval(j,x))) \
                            + np.dot(w[1]*self.eval(1,x), np.dot(c(x),self.eval(j,x))) \
                            + np.dot(w[2]*self.eval(2,x), np.dot(c(x),self.eval(j,x)))
                A[j] = self.integrate(f)
        else:
            for j in xrange(3):
                f = lambda x: np.dot(w[0]*self.eval(0,x), np.dot(c,self.eval(j,x))) \
                            + np.dot(w[1]*self.eval(1,x), np.dot(c,self.eval(j,x))) \
                            + np.dot(w[2]*self.eval(2,x), np.dot(c,self.eval(j,x)))
                A[j] = self.integrate(f)
        return np.array([A])


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

    def __mul__(self, other):
        """ multiplication of two elements. They should be defined on same mesh """
        method = self.method() + other.method(); deg = self.deg() + other.deg()
        return Element(method, deg)

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




