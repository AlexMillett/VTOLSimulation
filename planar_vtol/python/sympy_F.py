
#%%
import sympy as sp
from IPython.display import Math, display
from sympy import sin, cos, diff, Matrix, symbols, Function, pretty_print, simplify, latex, init_printing
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vpprint, vlatex

def F_doubleDots():
    t, mc, mr, ml, d, mu, g, fr, fl= symbols('t, mc, mr, ml, d, mu, g, fr, fl')
    z = Function('z')(t)
    theta = Function('theta')(t)
    h = Function('h')(t)

    #defining generalized coords and derivatives
    q = Matrix([[z], [theta], [h]])
    qdot = diff(q, t)  #q.diff(t)

    #defining the kinetic energy
    pc = Matrix([[z], [h], [0]])
    pr = Matrix([[z+(d*sp.cos(theta))], [h+(d*sp.sin(theta))], [0]])
    pl = Matrix([[z-(d*sp.cos(theta))], [h-(d*sp.sin(theta))], [0]])

    

    vc = diff(pc, t)
    vr = diff(pr, t)
    vl = diff(pl, t)


    omega = Matrix([[0], [0], [diff(theta,t)]])
    J = .0042
    
    K = simplify(0.5*mc*vc.T*vc + 0.5*mr*vr.T*vr + 0.5*ml*vl.T*vl+ 0.5*omega.T*J*omega)
    K = K[0,0]

    print("\n\n\n\n Kinetic energy for case study B (compare with book)")
    display(K)

    
    #defining potential energy (MUST BE A MATRIX as well to do L = K-P)
    P = (mc*g*h) + (mr*g*(h+d*sp.sin(theta))) + (mr*g*(h-d*sp.sin(theta)))

    #calculate the lagrangian, using simplify intermittently can help the equations to be
    #simpler, there are also options for factoring and grouping if you look at the sympy
    #documentation.

    L = simplify(K-P)

    
    # Solution for Euler-Lagrange equations, but this does not include right-hand side (like -B*q_dot and tau)
    EL_F = simplify( diff(diff(L, qdot), t) - diff(L, q) )


    
    zd = z.diff(t)
    zdd = zd.diff(t)
    thetad = theta.diff(t)
    thetadd = thetad.diff(t)
    hd = h.diff(t)
    hdd = hd.diff(t)

    RHS = Matrix([[-mu*zd - fl*d*cos(theta) - fr*d*cos(theta)], [(fl*d +fr*d)], [fl*d*sin(theta) - fr*d*sin(theta)]])


    print(sp.shape(EL_F))
    print(sp.shape(RHS))

    full_eom = EL_F - RHS

    display(Math(vlatex(full_eom)))

    
    # finding and assigning zdd and thetadd
    # print(set(symbols))
    result = simplify(sp.solve(full_eom, (zdd, thetadd, hdd)))

    # result is a Python dictionary, we get to the entries we are interested in
    # by using the name of the variable that we were solving for
    zdd_eom = result[zdd]  # EOM for zdd, as a function of states and inputs
    thetadd_eom = result[thetadd]
    hdd_eom = result[hdd]
    # EOM for thetadd, as a function of states and inputs

    display(Math(vlatex(zdd_eom)))
    
    display(Math(vlatex(thetadd_eom)))
    
    display(Math(vlatex(hdd_eom)))
    return zdd_eom, thetadd_eom, hdd_eom

    # %%
