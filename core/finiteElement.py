import numpy as np
from gaussquad import gaussPts
from shapefuncs import ShapeFunctions


class Element:
    """ Finite Element class
    Input:
            deg - degree of piecewise polynomial (int)
            gauss - degree of Gaussian quadrature (int)
    """
    def __init__(self, deg, gauss):
        self.__deg = deg
        self.__sfns = ShapeFunctions(self.__deg)
        self.__initialized = False
        self.__vertices = None 
        self.__dofs = None
        self.__jacobi_dets = None

        # get Gaussian quadrature points and weights ([x,y,w])
        xw = np.array(gaussPts(gauss, dim=2))

        self.__n_quad = xw.shape[0]             # number of quad points
        self.__quad_weights = xw[:,2]/2         # quad weights
        self.__quad_points = xw[:,:2]           # quad points

    def deg(self):
        """ return degree of element """
        return self.__deg

    def n_dofs(self):
        """ return number of dofs """
        return self.__sfns.n_dofs()

    def initialized(self):
        """ True if element vertices and DOFs are initialized """
        return self.__initialized

    def n_quad(self):
        """ return number of quadrature points """
        return self.__n_quad

    def quad_weights(self, n=None):
        """ return quadrature weights or weight of quad point n"""
        if n is None:
            return self.__quad_weights
        else:
            return self.__quad_weights[n]

    def quad_points(self, n=None):
        """ return quadrature points or quad point n"""
        if n is None:
            return self.__quad_points
        else:
            return self.__quad_points[n,:]

    def set_data(self, vertices, dofs=None):
        """Set data for element
        Input:
            vertices    - vertex coords of element
            dofs        - DOF coords of elememt, must correspond to degree (optional if degree equals 1)
        """
        if dofs is None: 
            dofs = vertices
        assert dofs.shape == (self.n_dofs(),2), "DOFs incompatible with shape functions."
        assert vertices.shape == (3,2), "Invalid vertex data."

        self.__vertices = vertices
        self.__dofs = dofs

        # set Jacobi determinant of transformation xi -> x, i.e. dx/dxi at quad points 
        jacobis = np.zeros((self.n_quad(),2,2))
        
        for j in range(self.n_dofs()):
            dphi_x, dphi_y = self.eval(j, self.quad_points(), derivative=True).T
            x, y = dofs[j,:]
            jacobis[:,0,0] += dphi_x*x
            jacobis[:,0,1] += dphi_y*x
            jacobis[:,1,0] += dphi_x*y
            jacobis[:,1,1] += dphi_y*y
            #print(jacobis); print("------------------------------")    
            #print(self.eval(j, self.quad_points(), derivative=True)); print("------------------------------")    
        self.__jacobi_dets = np.linalg.det(jacobis)
        self.__ijacobis = np.linalg.inv(jacobis)
        self.__initialized = True

    def clear_data(self):
        """ clear element data, i.e., vertices and DOFs """
        self.__vertices = None
        self.__dofs = None 
        self.__jacobi_dets = None 
        self.__initialized = False

    def vertices(self, n=None):
        """ return vertex coords or coord of vertex n """
        if n is None or not self.initialized():
            return self.__vertices
        else:
            return self.__vertices[n,:]

    def dofs(self, n=None):
        """ return DOF coords or coord of DOF n """
        if n is None or not self.initialized():
            return self.__dofs
        else:
            return self.__dofs[n,:]

    def jacobi_dets(self):
        """ return jacobi dets of transformation xi -> x at quad points 
        with P1 basis det = 2*measure """
        return self.__jacobi_dets

    def measure(self):
        """ returns measure (area) of element """
        if self.initialized():
            return np.linalg.det(np.concatenate((self.vertices(),np.ones((3,1))),axis=1))/2
        else:
            return 0

    def eval(self, n, xi, derivative=False):
        """Evaluate shape function n (or derivative) at ref coord xi
        Input:
            n   - n'th shape function to be evaluated
            xi  - local coord
        """
        return self.__sfns.eval(n, xi, derivative=derivative)

    def map_to_elt(self, xi):
        """ maps ref coord xi to global coord x """
        x = np.sum([self.eval(j, xi)*self.dofs(j)[0] for j in range(self.n_dofs())], axis=0)
        y = np.sum([self.eval(j, xi)*self.dofs(j)[1] for j in range(self.n_dofs())], axis=0)
        return np.array([x,y]).T

    def integrate(self, f):
        """ integrate (global) scalar function f over element
        Input:
            f - function to be integrated (callable or constant)
        Output:
            integral
        """
        if callable(f):
            integrand = f(self.map_to_elt(self.quad_points()))
            return sum(integrand*self.jacobi_dets()*self.quad_weights())
        else:
            return f*self.measure()
                
    #def assemble(self, c=None, derivative=True):
    def assemble_P1_stiffness(self):
        """ shortcut method for assembling stiffness matrix for P1 elemenets """
        v1,v2,v3 = self.vertices()
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3
        D = np.array([[x3-x2, x1-x3, x2-x1],[y3-y2, y1-y3, y2-y1]])
        return np.dot(D.T,D)/(4*self.measure())

        #E = self.__vertices[[1, 2, 0],:] - self.__vertices[[2, 0, 1],:]
        #return np.dot(E,E.T)/(4*self.measure())

    def assemble(self, c=None, derivative=True):
        """Assembles local stiffness (default) or mass matrix 
        Input:
            c               - coefficient (variable or constant), can be tensor valued if derivative
            derivative      - True if stiffness, false if mass
        Output:
            A               - matrix of (c dphi_i, dphi_j) or (c phi_i, phi_j)
        """
        n = self.n_dofs()
        A = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                A[i,j] = self.integrate_phi_phi(i, j, c=c, derivative=derivative)
        return A

    def assemble_rhs(self, f):
        """Assembles local right hand side
        Input:
            f       - function (callable or constant)
        Output:
            rhs     - vector of (phi_i, f)
        """
        n = self.n_dofs()
        rhs = np.zeros(n)
        for i in range(n):
            rhs[i] = self.integrate_phi_f(i, f)
        return rhs

    def integrate_phi_phi(self, i, j, c=None, derivative=False):
        """Integrate product of shape functions i and j product over element """

        if derivative:
            #ix, iy = self.eval(i, self.quad_points(), derivative).T
            #jx, jy = self.eval(j, self.quad_points(), derivative).T
            gradi = self.eval(i, self.quad_points(), derivative)
            gradj = self.eval(j, self.quad_points(), derivative)
            if c is None:
                #integrand = ix*jx + iy*jy; 
                integrand = np.zeros(self.n_quad())
                inv_jaco = self.__ijacobis
                for k in range(self.n_quad()):
                    tempi = np.dot(gradi[k,:], inv_jaco[k,:,:])
                    tempj = np.dot(gradj[k,:], inv_jaco[k,:,:])
                    integrand[k] = np.dot(tempi,tempj)

                #print(ix);print(iy);print(jx);print(jy)
                #print(integrand)
                
            else:
                if callable(c): # <============================ variable coefficient
                    c_eval = c(self.quad_points())
                    if c_eval.ndim == 3: # <------------------- tensor valued 
                        integrand = c_eval[:,1,1]*ix*jx + c_eval[:,1,2]*ix*jy \
                                  + c_eval[:,2,1]*iy*jx + c_eval[:,2,2]*iy*jx
                    else: # <---------------------------------- scalar valued 
                        integrand = c_eval*(ix*jx + iy*jy)
                else: # <====================================== constant coefficient
                    if len(c) == 2: # <------------------------ tensor 
                        integrand = c[1,1]*ix*jx + c[1,2]*ix*jy \
                                  + c[2,1]*iy*jx + c[2,2]*iy*jx
                    else: # <---------------------------------- scalar 
                        integrand = c*(ix*jx + iy*jy)
            #integrand *= self.__ijacobi_dets**2
        else:
            phi_i = self.eval(i, self.quad_points(), derivative)
            phi_j = self.eval(j, self.quad_points(), derivative)
            if c is None:
                integrand = phi_i*phi_j
            else:
                if callable(c):
                    integrand = phi_i*phi_j*c(self.quad_points())
                else:
                    integrand = c*phi_i*phi_j
        #print(self.jacobi_dets())
        #print(self.measure())
        #print(self.quad_weights())
        #print(integrand*self.jacobi_dets()*self.quad_weights())
        #print("========================================")
        return sum(integrand*self.jacobi_dets()*self.quad_weights())#/(2*self.measure())
                    
    def integrate_phi_f(self, i, f):
        """Integrate product of shape function i and function f """
        if callable(f):
            integrand = self.eval(i, self.quad_points(), derivative=False)*f(self.quad_points())
        else:
            integrand = self.eval(i, self.quad_points(), derivative=False)*f
        return sum(integrand*self.jacobi_dets()*self.quad_weights())


#--------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------#

if __name__ == "__main__":
    """ Here we do some tests """
        
    def test1():
        from meshing import RectangleMesh
        mesh = RectangleMesh(nx=1,ny=1)
        fe = Element(1,4)
        for j in range(mesh.n_elts()):
            vs = mesh.elt_to_vcoords(j)
            fe.set_data(vs)
            print("\n--------- element {} ------------".format(j))
            print("measure: \t\t", fe.measure())
            print("det: \t\t\t", fe.jacobi_dets())
            print("assemble stiffness:\n{}".format(fe.assemble()))
            print("assemble mass:\n{}".format(fe.assemble(derivative=False)))
            print("assemble rhs: {}".format(fe.assemble_rhs(1)))
            #print("map to elt:\n{}".format(fe.map_to_elt(np.ones((4,2)))))
            #print("map to elt:\n{}".format(fe.map_to_elt(np.array([0,0]))))

    def map_test():
        from meshing import RectangleMesh
        mesh = RectangleMesh(nx=3,ny=3,deg=2)
        ref_vertices = np.array([[0,1],[0,0],[1,0]])
        fe = Element(2,4)
        vertices = mesh.elt_to_vcoords(); dofs = mesh.elt_to_dofcoords()
        j = 0
        for v,d in zip(vertices, dofs):
            fe.set_data(v, d)
            mp = fe.map_to_elt(ref_vertices)
            print("============================\n")
            print("Vertex mapping:", np.allclose(fe.vertices(), mp))
            print("Center mapping:", np.allclose(fe.map_to_elt(np.array([1/3,1/3])), mesh.elt_to_ccoords(j)))
            j += 1

    def integral_test(deg=1,gauss=1):
        from meshing import RectangleMesh
        mesh = RectangleMesh(nx=4,ny=4,deg=deg,diag="l")
        #f = lambda x: 1                        # sum to 1
        #f = lambda x: np.sum(x,1)              # sum to 1
        #f = lambda x: np.prod(x,1)             # sum to 1/4
        f = lambda x: np.sin(np.prod(x,1))      # sum to 0.239812

        fe = Element(deg,gauss)
        vertices = mesh.elt_to_vcoords()
        dofs = mesh.elt_to_dofcoords()

        integral = 0
        for v,d in zip(vertices, dofs):
            fe.set_data(v,d)
            #print(fe.jacobi_dets())
            #print(fe.integrate(f))
            #print(fe.measure())
            #print("===================\n")
            integral += fe.integrate(f)
        print("Integral is:", integral)


    def assemble_test(n=4,gauss=2):
        from meshing import RectangleMesh
        mesh = RectangleMesh(nx=n,ny=n,diag="l")

        fe = Element(1,gauss)
        vertices = mesh.elt_to_vcoords()

        for v in vertices:
            fe.set_data(v)
            a1 = fe.assemble()
            a2 = fe.assemble_P1_stiffness()
            print(a1)
            print(a2)
            print(np.isclose(a1, a2))
            print("===================\n")
            
    
    #test1()       
    #map_test() 
    #integral_test(8,10)
    assemble_test(n=7)





