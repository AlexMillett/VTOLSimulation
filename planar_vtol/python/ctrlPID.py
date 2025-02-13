import numpy as np
import VTOLParam as P


class ctrlPID:
    def __init__(self):
        # dirty derivative parameters
        self.sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * self.sigma - P.Ts) / (2 * self.sigma + P.Ts)  

        ####################################################
        #       PD Control: Time Design Strategy
        ####################################################
        # tuning parameters
        tr_h = 3.0  # rise time for altitude - original (3.0)
        zeta_h = 0.95  # damping ratio for altitude - original (0.707)
        tr_z = 3.0  # rise time for outer lateral loop (position) - original
        M = 10.0  # time separation between inner and outer lateral loops
        zeta_z = 0.9  # damping ratio for outer lateral loop
        zeta_th = 0.9  # damping ratio for inner lateral loop
        self.ki_h = 0.10  # integrator on altitude

        # for ki_z, it actually works best if ki_z = 0.0!! This tells us something
        # about the system type. But it is possible to get reasonable performance
        # if we increase zeta for theta and z as shown below. The negative is
        # strange, but if we examine kp_z and kd_z, they are also negative. This
        # is because the plant itself has a negative gain. Making kz_i positive
        # actually drives the system to be unstable.
        self.ki_z = -0.005  # integrator on position

        # saturation limits
        self.theta_max = 10.0 * np.pi / 180.0  # Max theta, rads
        self.Fe = (P.mc + 2.0 * P.mr) * P.g  # equilibrium force   

        #---------------------------------------------------
        # PD gains for longitudinal (altitude) control
        wn_h = 2.2 / tr_h  # original equation for natural frequency (before changing zeta from 0.707)
        Delta_cl_d = [1, 2 * zeta_h * wn_h, wn_h ** 2.0]  # desired closed loop char eq
        self.kp_h = Delta_cl_d[2] * (P.mc + 2.0 * P.mr)  # kp = altitude
        self.kd_h = Delta_cl_d[1] * (P.mc + 2.0 * P.mr)  # kd = altitude

        #---------------------------------------------------
        # PD gains for lateral inner loop
        b0       = 1.0/(P.Jc+2.0*P.mr*P.d**2)
        tr_th    = tr_z/M
        wn_th    = 2.2/tr_th
        self.kp_th  = wn_th**2.0/b0
        self.kd_th  = 2.0*zeta_th*wn_th/b0

        #---------------------------------------------------
        #PD gain for lateral outer loop
        b1       = -self.Fe/(P.mc+2.0*P.mr)
        a1       = P.mu/(P.mc+2.0*P.mr)
        wn_z     = 2.2/tr_z
        self.kp_z   = wn_z**2.0/b1
        self.kd_z   = (2.0*zeta_z*wn_z-a1)/b1

        # print control gains to terminal        
        print('kp_z: ', self.kp_z)
        print('kd_z: ', self.kd_z)
        print('kp_h: ', self.kp_h)
        print('kd_h: ', self.kd_h)
        print('kp_th: ', self.kp_th)
        print('kd_th: ', self.kd_th)

        #---------------------------------------------------
        # initialize variables needed for integrator and differentiators
        #---------------------------------------------------
        self.integrator_h = 0.
        self.error_h_d1 = 0.
        self.h_dot = 0.
        self.h_d1 = 0.
        self.integrator_z = 0.
        self.error_z_d1 = 0.
        self.z_dot = 0.
        self.z_d1 = 0.
        self.theta_dot = 0.
        self.theta_d1 = 0.

    def update(self, r, y):
        z_r = r[0][0]
        h_r = r[1][0]
        z = y[0][0]
        h = y[1][0]
        theta = y[2][0]

        #---------------------------------------------------
        # Update altitude control
        #---------------------------------------------------
        # Compute the error in h
        error_h = h_r - h
        
        # differentiate h
        self.h_dot = self.beta * self.h_dot + (1 - self.beta) * ((h - self.h_d1) / P.Ts)

        # integrate error in h if h_dot is small. This is to prevent integrator.
        if np.abs(self.h_dot) < 0.5:
            self.integrator_h = self.integrator_h + (P.Ts / 2) * (error_h + self.error_h_d1)
        
        # PID control - unsaturated
        F_tilde = self.kp_h * (h_r - h) + self.ki_h * self.integrator_h - self.kd_h * self.h_dot
        F_unsat = F_tilde + self.Fe
        F = saturate( F_unsat, 2*P.max_thrust)
        
        #---------------------------------------------------
        # Update position control
        #---------------------------------------------------
        # Compute the error in z
        error_z = z_r - z
        
        # differentiate z
        self.z_dot = self.beta * self.z_dot + (1 - self.beta) * ((z - self.z_d1) / P.Ts)

        # integrate error in z if z_dot is small enough. This is to prevent integrator.
        if np.abs(self.z_dot) < 0.5:
            self.integrator_z = self.integrator_z + (P.Ts / 2) * (error_z + self.error_z_d1)

        # PID control - unsaturated
        theta_r = self.kp_z * error_z + self.ki_z * self.integrator_z - self.kd_z * self.z_dot
        
        # saturate theta_r
        #theta_r = saturate(theta_r_unsat, self.theta_max)
        
        #---------------------------------------------------
        # Update pitch control
        #---------------------------------------------------
        # differentiate theta
        self.theta_dot = self.beta * self.theta_dot + (1 - self.beta) * ((theta - self.theta_d1) / P.Ts)
        
        # PD control on theta
        tau = saturate( self.kp_th * (theta_r - theta) - self.kd_th * self.theta_dot, 2*P.max_thrust*P.d)
        
        # update delayed variables
        self.error_h_d1 = error_h
        self.h_d1 = h
        self.error_z_d1 = error_z
        self.z_d1 = z
        self.theta_d1 = theta
        motor_thrusts = P.mixing @ np.array([[F], [tau]])

        return motor_thrusts
 

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

 
