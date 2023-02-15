def test_ex_2b():

    import numpy as np 
    from nm_lib import nm_lib as nm


    x0 = -2.6
    xf = 2.6 

    nint = 64
    nump = nint + 1

    x = np.linspace(x0, xf, nump)

    def initial_u(xx):

        r"""
        Computing the initial condition of u_i
        
        Requires
        ----------
        Numpy

        Parameters
        ----------
        xx : `array`
            Spatial array

        Returns
        ------- 
        initial condition : `array`
            initial condition of u_i
        """

        return np.cos(6*np.pi*xx/5)**2/np.cosh(5*xx**2)
    
    u0 = initial_u(x)


    def analytical(xx, nt, a, t): 

        r"""
        Computes the analytical solution of Eq.(1)
        
        Requires
        ----------
        Numpy

        Parameters
        ----------
        xx : `array`
            spatial array.
        nt : `array`
            number of time steps
        a : `float` or `array`
            Either constant, or array which multiply the right hand side of the Burger's eq.
        t : `array`
            time

        Returns
        ------- 
        A : `array`
            analytical solution with periodic boundaries
        """

        A = np.zeros((len(xx), nt))
        U = np.zeros((len(xx), nt))

        for i in range(nt):
            U[:,i] = (xx-a*t[i])

            """
            The -a*t/5.2)[-1], in the following range is to round up how many times the 
            function "passes the wall". This way we know how many times we need to
            wrap the analytical function
            """
            for j in reversed(range(int(np.round(-a*t/5.2)[-1]))): 
                U[np.where(U[:,i] > 2.6+j*5.2)[0], i]  = U[np.where(U[:,i] > 2.6 + 5.2*j)[0], i] - (j+1)*5.2

            A[:,i] = initial_u(U[:,i])

        return A
    
    nt = 400
    a = -1

    t, un = nm.evolv_adv_burgers(x , u0 , nt ,a, ddx=nm.deriv_dnw, bnd_limits=[0,1])

    A = analytical(x, nt, a, t)

    time_  = np.argmin(np.abs(t -12))
    peak = np.argmax(un [:,time_ ])

    diff_64 = np.abs(un[peak, time_] - A[peak, time_])

    if diff_64 < 0.25:
        assert True
    else: 
        assert False
