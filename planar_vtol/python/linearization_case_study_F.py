#%%
from sympy import sin, cos, Matrix, symbols, init_printing, latex
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vpprint, vlatex
from IPython.display import Math, display

#defining mathematical variables (called symbols in sympy) and time varying functions like z and theta and h
t, mc, mr, Jc, d, mu, g, F, tau = symbols('t, m_c, m_r, J_c, d, mu, g, F, tau ')
z = dynamicsymbols('z')
h = dynamicsymbols('h')
theta = dynamicsymbols('theta')

# Calculating the mass matrix first, it would be nice to be able to factor this out automatically, rather than
# having to type it explicitly and then invert, but ... c'est la vie for right now.
# If we had derived these equations like we show in the example for case study B, we could
# just solve directly for the second derivatives of the generalized coordinates instead. 
M = Matrix([[mc+2*mr, 0, 0], [0, mc+2*mr, 0], [0, 0, Jc+2*mr*d**2]])

# forming the right-hand side (RHS) of the equations of motion to get state variable form
RHS = Matrix([[-F*sin(theta)-mu*z.diff(t)], [-(mc+2*mr)*g+F*cos(theta)], [tau]])

EOM_dd = M.inv()*RHS
zdd = EOM_dd[0]
hdd = EOM_dd[1]
thetadd = EOM_dd[2]

# this should be the equations of motion in state variable form
f_of_x_and_u = Matrix([[z.diff(t)], [h.diff(t)], [theta.diff(t)], [zdd], [hdd], [thetadd]])
print("EOM, solved for second derivatives of generalized coordinates :\n")
display(Math(vlatex((f_of_x_and_u))))


# defining states and inputs symbolically
states = (Matrix([z, h, theta, z.diff(t), h.diff(t), theta.diff(t)])).reshape(6,1)
inputs = (Matrix([F, tau])).reshape(2,1) 

# finding the jacobian with respect to states (A) and inputs (B)
A = f_of_x_and_u.jacobian(states)
B = f_of_x_and_u.jacobian(inputs)

# substituting in the equilibrium values for each state and input (finding the equilibrium points can likely be
# done automatically in sympy as well, but we are currently defining them by hand)
A_lin = A.subs([(theta.diff(t),0), (theta, 0), (F, (mc+2*mr)*g), (tau, 0)])
B_lin = B.subs([(theta.diff(t),0), (theta, 0), (F, (mc+2*mr)*g), (tau, 0)])

print("Linearized A Matrix is :\n")
display(Math(vlatex(A_lin)))

print("Linearized B Matrix is :\n")
display(Math(vlatex(B_lin)))

# Picking off the entries corresponding to h and h_dot and F for input.
# We can only do this because the A matrix shows that h and h_dot for the 
# longitudinal dynamids are not affected by the other states (or lateral dynamics).  
A_lin_lon = A_lin[[1, 4], [1, 4]]
B_lin_lon = B_lin[[1, 4], [0]]

print("Linearized A_lon Matrix is :\n")
display(Math(vlatex(A_lin_lon)))

print("Linearized B_lon Matrix is :\n")
display(Math(vlatex(B_lin_lon)))


# Now picking off the entries corresponding to z, theta, z_dot, and theta_dot 
# for the lateral dynamics
A_lin_lat = A_lin[[0, 2, 3, 5], [0, 2, 3, 5]]
B_lin_lat = B_lin[[0, 2, 3, 5], [1]]

print("Linearized A_lat Matrix is :\n")
display(Math(vlatex(A_lin_lat)))

print("Linearized B_lat Matrix is :\n")
display(Math(vlatex(B_lin_lat)))
# %%
