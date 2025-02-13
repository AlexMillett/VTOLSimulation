
#%%
import sympy as sp
from IPython.display import Math, display
from sympy import sin, cos, diff, Matrix, symbols, Function, pretty_print, simplify, latex, init_printing
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vpprint, vlatex


t, mc, mr, ml, d, mu, g, s = symbols('t, m_c, m_r, m_l, d, mu, g, s')
z = dynamicsymbols('z')
theta = dynamicsymbols('theta')
h = dynamicsymbols('h')

#defining generalized coords and derivatives
q = Matrix([[z], [theta], [h]])
qdot = q.diff(t)  #q.diff(t)

#defining the kinetic energy
pc = Matrix([[z], [h], [0]])
pr = Matrix([[z+(d*sp.cos(theta))], [h+(d*sp.sin(theta))], [0]])
pl = Matrix([[z-(d*sp.cos(theta))], [h-(d*sp.sin(theta))], [0]])



vc = diff(pc, t)
vr = diff(pr, t)
vl = diff(pl, t)


omega = Matrix([[0], [0], [diff(theta,t)]])
J = sp.symbols('J')

K = simplify(sp.Rational(1,2)*(mc*vc.T@vc + mr*vr.T@vr + mr*vl.T@vl+ J*omega.T@omega))
K = K[0,0]

print("\n\n\n\n Kinetic energy for case study B (compare with book)")
display(Math(vlatex(K)))

#%%
#defining potential energy (MUST BE A MATRIX as well to do L = K-P)
P = (mc*g*h) + (mr*g*(h+d*sp.sin(theta))) + (mr*g*(h-d*sp.sin(theta)))

#calculate the lagrangian, using simplify intermittently can help the equations to be
#simpler, there are also options for factoring and grouping if you look at the sympy
#documentation.

L = simplify(K-P)


# Solution for Euler-Lagrange equations, but this does not include right-hand side (like -B*q_dot and tau)
EL_F = simplify( diff(diff(L, qdot), t) - diff(L, q) )

print('EL_F')
display(Math(vlatex(EL_F)))

#%%
zd = z.diff(t)
zdd = zd.diff(t)
thetad = theta.diff(t)
thetadd = thetad.diff(t)
hd = h.diff(t)
hdd = hd.diff(t)
F, torque = sp.symbols("F tau")

RHS = Matrix([[-mu*zd - F*sin(theta)], [torque], [F*cos(theta) - g*(mc+(2*mr))]])
# print("RHS")
# display(Math(vlatex(RHS)))

full_eom = EL_F - RHS

display(Math(vlatex(full_eom)))

#%%
# finding and assigning zdd and thetadd
# print(set(symbols))
result = simplify(sp.solve(full_eom, (zdd, thetadd, hdd)))

# result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
zdd_eom = result[zdd]  # EOM for zdd, as a function of states and inputs
thetadd_eom = result[thetadd]
hdd_eom = result[hdd]
# EOM for thetadd, as a function of states and inputs
# print("zdd_eom")
# display(Math(vlatex(zdd_eom)))
# print("thatadd_eom")
# display(Math(vlatex(thetadd_eom)))
# print("hdd_eom")
# display(Math(vlatex(hdd_eom)))

#%%

state_variable_form = Matrix([[zdd_eom], [zd], [thetadd_eom], [thetad], [hdd_eom], [hd]])
states = Matrix([[zd], [z], [thetad], [theta], [hd], [h]])
inputs = Matrix([[F], [torque]])

#%%
# finding the jacobian with respect to states (A) and inputs (B)
A = state_variable_form.jacobian(states)
B = state_variable_form.jacobian(inputs)

zero_zdd = zdd_eom.subs([(thetad, 0),(zd, 0),(hd, 0)])
print("zero_zdd")
display(Math(vlatex(zero_zdd)))
zero_thetadd = thetadd_eom.subs([(thetad, 0),(zd, 0),(hd, 0)])
print("zero_thetadd")
display(Math(vlatex(zero_thetadd)))
zero_hdd = hdd_eom.subs([(thetad, 0),(zd, 0),(hd, 0)])
print("zero_hdd")
display(Math(vlatex(zero_hdd)))
#%%
display(Math(vlatex(state_variable_form)))
#%%
result1 = sp.solve([zero_zdd], (theta))
display(Math(vlatex(result1)))
result2 = sp.solve(zero_thetadd, (torque))
display(Math(vlatex(result2)))
result3 = sp.solve(zero_hdd, (F))
display(Math(vlatex(result3)))

theta_e = result1[0][0]
torque_e = result2[0]
F_e = result3[0]
F_e = F_e.subs([(theta, theta_e)])

z_e, h_e = sp.symbols('z_e,h_e')

#%%
# sub in values for equilibrium points (x_e, u_e) or (x_0, u_0)
A_lin = simplify(A.subs([(zd,0.), (z,z_e),(thetad,0.), (theta,theta_e), (hd,0.), (h,h_e),(F, F_e)]))
B_lin = simplify(B.subs([(zd,0.), (z,z_e),(thetad,0.), (theta,theta_e), (hd,0.), (h,h_e),(F, F_e)]))

display(Math(vlatex(A_lin)))
display(Math(vlatex(B_lin)))
# %%

feedback_zdd_eom = zdd_eom.subs([(F, F + F_e)])
print("Feedback linearization zdd")
display(Math(vlatex(feedback_zdd_eom)))


feedback_thetadd_eom = thetadd_eom.subs([(F, F+F_e)])
print("Feedback linearization theta dd")
display(Math(vlatex(feedback_thetadd_eom)))

feedback_hdd_eom = hdd_eom.subs([(F, F + F_e)])
print("Feedback linearization h dd")
display(Math(vlatex(feedback_hdd_eom)))

# %%
# We need matricies C and D
C_lin = Matrix([[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]])
D_lin = Matrix([[0,0,0,0,0,0],[0,0,0,0,0,0]]).transpose()

# This converts our linear function matricies A,B,C, and D into the transfer function
x_fer = simplify(C_lin @ (((s * sp.eye(A_lin.shape[0])) - A_lin).inv()) @ (B_lin + D_lin))
display(Math(vlatex(x_fer)))

# %%
