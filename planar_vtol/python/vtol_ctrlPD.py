
import numpy as np
import VTOLParam as P


class ctrlPD:

    def __init__(self):
    # tuning parameters

        tr_h = 4.0 # rise time for altitude - tuned for best rise time without saturation
        zeta_h = 0.707  # damping ratio for altitude
        tr_z = 4.0 # rise time for outer lateral loop (position) - tuned for best rise time without saturation
        zeta_z = 0.707 # damping ratio for outer lateral loop
        zeta_th = 0.707 # damping ratio for inner lateral loop

    # saturation limits
        self.theta_max = 10.0 * np.pi / 180.0 # Max theta, rads
        self.Fe = (P.mc + 2.0 * P.mr) * P.g # equilibrium force

    # Finding PD gains for longitudinal (altitude) control
        wn_h = 2.2/tr_h # natural frequency
        Delta_cl_d = [1, 2*zeta_h*wn_h, wn_h**2.0] # desired closed loop char eq
        self.kp_h = Delta_cl_d[2]*(P.mc+2.0*P.mr) # kp - altitude
        self.kd_h = Delta_cl_d[1]*(P.mc+2.0*P.mr) # kd = altitude


    # Finding PD gains for lateral inner loop
        M = 10.0 # time separation between inner and outer lateral loops
        b0 = 1.0/(P.Jc+2.0*P.mr*P.d**2)
        tr_th = tr_z/M
        wn_th = 2.2/tr_th
        self.kp_th = wn_th**2.0/b0
        self.kd_th = 2.0*zeta_th*wn_th/b0

    # Finding PD gain for lateral outer loop
        b1 = -self.Fe/(P.mc+2.0*P.mr)
        a1 = P.mu/(P.mc+2.0*P.mr)
        wn_z = 2.2/tr_z
        self.kp_z = wn_z**2.6/b1
        self.kd_z = (2.0*zeta_z*wn_z-a1)/b1


        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)
        print('kp_h: ', self.kp_h)
        print('kd_h: ', self.kd_h)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)

    def update(self, reference, state):
        z_r = reference[0][0]
        h_r = reference[1][0]
        z = state[0][0]
        h = state[1][0]
        theta = state[2][0]
        zdot = state[3][0]
        hdot = state[4][0]
        thetadot = state[5][0]

    # calculate control for the altitude loop

        F_tilde = self.kp_h * (h_r - h) - self.kd_h * hdot
        F = saturate( F_tilde + self.Fe, 2*P.fmax)

        # calculate theta_r from the outer loop
        theta_r = saturate( self.kp_z * (z_r - z) - self.kd_z * zdot, self.theta_max)

        # calculate torque from the inner loop
        tau = saturate(self.kp_th * (theta_r - theta) - self.kd_th * thetadot, 2*P.fmax*P.d)

        # convert force and torque to motor thrusts/forces
        motor_thrusts = P.mixing @ np.array([[F], [tau]])

        return motor_thrusts

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u
