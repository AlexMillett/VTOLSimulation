#%%
from linearization_case_study_F import *
from sympy import eye, zeros, simplify

# the order for rows in C is just because I had defined my states as [z, h, theta, z_dot, h_dot, theta_dot], but I want
# h first, then z, then theta for my transfer functions
C = Matrix([[0, 1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0]])
D = Matrix(zeros(3,2))

#%%
# these are the three transfer functions for h, z, and theta with respect to inputs F and tau
s = symbols('s')
transfer_func = simplify(C@(s*eye(6)-A_lin).inv() @B_lin+D)

#%%
# these indices that we select from transfer_func are based on the 
# output (row) and input (column) 
print("\nTransfer function H(s)/F(s)")
display(Math(vlatex(transfer_func[0, 0])))

print("\n\n\nTransfer function Z(s)/Tau(s)")
display(Math(vlatex(transfer_func[1, 1])))

print("\n\n\nTransfer function Theta(s)/Tau(s)")
display(Math(vlatex(transfer_func[2, 1])))


# %%
