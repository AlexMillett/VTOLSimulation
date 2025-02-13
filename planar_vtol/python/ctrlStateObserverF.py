import numpy as np
import VTOLParam as P
import control as cnt
from scipy import signal

class ctrl:
    def __init__(self):
        # initialize wind force (this term was always in the dynamics solution, but may not be
        # included in your own, so please check).
        P.F_wind = 0.1

        # tuning parameters 
        tr_h = 10
        tr_th = .1
        tr_z = 10

        wn_h = 2.2/tr_h
        zeta_h = 0.707
        wn_z = 2.2/tr_z
        zeta_z = 0.707
        wn_th = 2.2/tr_th
        zeta_th = 0.707

        integrator_h = -1.0
        integrator_z = -3.0

        tr_h_obs = 1
        tr_th_obs = 1
        tr_z_obs = 1

        

        # State Space Equations
        self.Fe = (P.mc + 2.0 * P.mr) * P.g  # equilibrium force 
        self.A_lon = np.array([[0.0, 1.0],
                        [0.0, 0.0]])
        self.B_lon = np.array([[0.0],
                        [1.0 / (P.mc + 2.0 * P.mr)]])
        self.C_lon = np.array([[1.0, 0.0]])
        self.A_lat = np.array([[0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, -self.Fe / (P.mc + 2.0 * P.mr), -(P.mu / (P.mc + 2.0 * P.mr)), 0.0],
                        [0.0, 0.0, 0.0, 0.0]])
        self.B_lat = np.array([[0.0],
                        [0.0],
                        [0.0],
                        [1.0 / (P.Jc + 2 * P.mr * P.d ** 2)]])
        self.C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])
        
        # form augmented system
        A1_lon = np.vstack((
                np.hstack((self.A_lon, np.zeros((2,1)))),
                np.hstack((-self.C_lon, np.zeros((1,1))))))
        B1_lon = np.vstack((self.B_lon, np.zeros((1,1))))
        A1_lat = np.vstack((
                np.hstack((self.A_lat, np.zeros((4,1)))),
                np.hstack((-self.C_lat[0:1], np.zeros((1,1))))))
        B1_lat = np.vstack((self.B_lat, np.zeros((1,1))))

        # gain calculation
        des_char_poly_lon = np.convolve([1.0, 2.0 * zeta_h * wn_h, wn_h ** 2],
                                        [1, integrator_h])
        
        des_poles_lon = np.roots(des_char_poly_lon)

        des_char_poly_lat = np.convolve(
            np.convolve([1.0, 2.0 * zeta_z * wn_z, wn_z ** 2],
                        [1.0, 2.0 * zeta_th * wn_th, wn_th ** 2]),
            [1, integrator_z])
        
        des_poles_lat = np.roots(des_char_poly_lat)

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A1_lon, B1_lon)) != 3:
            print("The longitudinal system is not controllable")
        else:
            K1_lon = cnt.place(A1_lon, B1_lon, des_poles_lon)
            self.K_lon = K1_lon[0][0:2]
            self.ki_lon = K1_lon[0][2]
        if np.linalg.matrix_rank(cnt.ctrb(A1_lat, B1_lat)) != 5:
            print("The lateral system is not controllable")
        else:
            K1_lat = cnt.place(A1_lat, B1_lat, des_poles_lat)
            self.K_lat = K1_lat[0][0:4]
            self.ki_lat = K1_lat[0][4]

        wn_h_obs = 2.2/tr_h_obs
        wn_z_obs = 2.2/tr_z_obs
        wn_th_obs = 2.2/tr_th_obs

        des_obs_lat_char_poly = np.convolve(
                [1, 2 * zeta_z * wn_z_obs, wn_z_obs**2],
                [1, 2 * zeta_th * wn_th_obs, wn_th_obs**2])
        des_obs_lat_poles = np.roots(des_obs_lat_char_poly)

        des_obs_lon_char_poly = [1, 2*zeta_h*wn_h_obs, wn_h_obs**2]
        des_obs_lon_poles = np.roots(des_obs_lon_char_poly)
        # Compute the gains if the system is controllable

        if np.linalg.matrix_rank(cnt.ctrb(self.A_lon.T, self.C_lon.T)) != 2:
            print("The system is not observerable")
        else:
            self.L_lon = cnt.acker(self.A_lon.T, self.C_lon.T, des_obs_lon_poles).T


        if np.linalg.matrix_rank(cnt.ctrb(self.A_lat.T, self.C_lat.T)) != 4:
            print("The system is not observable")
        else:
            self.L_lat = signal.place_poles(self.A_lat.T, self.C_lat.T, 
                                        des_obs_lat_poles).gain_matrix.T

        print('K_lon: ', self.K_lon)
        print('ki_lon: ', self.ki_lon)
        print('K_lat: ', self.K_lat)
        print('ki_lat: ', self.ki_lat)

        self.integrator_z = 0.0  # integrator on position z
        self.error_z_d1 = 0.0  # error signal delayed by 1 sample
        self.integrator_h = 0.0  # integrator on altitude h
        self.error_h_d1 = 0.0  # error signal delayed by 1 sample

        # this is a bit clunky, but is an attempt to handle saturation
        # limits without having to define them as a function of theta
        # (which they probably are) or in the motor/thrust frame.
        self.x_hat = np.array([
            [0.0],  # initial estimate for z_hat
            [0.0],
            [0.0],  # initial estimate for z_hat
            [0.0],  # initial estimate for theta_hat
            [0.0],  # initial estimate for z_hat_dot
            [0.0]])

        self.F_limit = P.max_thrust * 2.0
        self.tau_limit = P.max_thrust * P.d * 2.0

        self.F_d1 = 0.0 
        self.Tau_d1 = 0



    def update(self, r, y):

        x_hat = self.update_observer(y)
        z_hat = x_hat[0][0]
        h_hat = x_hat[1][0]
        th_hat = x_hat[2][0]

        z_r = r[0]
        h_r = r[1]

        # integrate error
        error_z = z_r - z_hat
        self.integrator_z += (P.Ts/2.0)*(error_z + self.error_z_d1)
        self.error_z_d1 = error_z

        error_h = h_r - h_hat
        self.integrator_h += (P.Ts/2.0)*(error_h + self.error_h_d1)
        self.error_h_d1 = error_h

        # Construct the states
        x_lon = np.array([[x_hat[1][0]], [x_hat[4][0]]])
        x_lat = np.array([[x_hat[0][0]], [x_hat[2][0]], [x_hat[3][0]], [x_hat[5][0]]])

        # Compute the state feedback controllers
        F_tilde = -self.K_lon @ x_lon - self.ki_lon * self.integrator_h
        F = self.Fe / np.cos(y[2][0]) + F_tilde[0]
        F_sat = saturate(F, self.F_limit)
        self.integratorAntiWindup(F_sat, F, self.ki_lon, self.integrator_h)

        tau = -self.K_lat @ x_lat - self.ki_lat*self.integrator_z
        tau_sat = saturate(tau[0], self.tau_limit)
        self.integratorAntiWindup(tau_sat, tau, self.ki_lat, self.integrator_z)

        self.F_d1 = F_sat
        self.Tau_d1 = tau_sat
       

        return np.array([[F_sat], [tau_sat]]), x_hat
    
    def update_observer(self, y_m):
        # update the observer using RK4 integration
        y_lon = np.array([[y_m[1][0]]])
        y_lat = np.array([[y_m[0][0]],[y_m[2][0]]])

        x_lon = np.array([[self.x_hat[1][0]], [self.x_hat[4][0]]])
        x_lat = np.array([[self.x_hat[0][0]], [self.x_hat[2][0]], [self.x_hat[3][0]], [self.x_hat[5][0]]])

        F1_lon = self.observer_f_lon(x_lon, y_lon)
        F2_lon = self.observer_f_lon(x_lon + P.Ts / 2 * F1_lon, y_lon)
        F3_lon = self.observer_f_lon(x_lon + P.Ts / 2 * F2_lon, y_lon)
        F4_lon = self.observer_f_lon(x_lon + P.Ts * F3_lon, y_lon)
        x_lon += P.Ts / 6 * (F1_lon + 2 * F2_lon + 2 * F3_lon + F4_lon)

        F1_lat = self.observer_f_lat(x_lat, y_lat)
        F2_lat = self.observer_f_lat(x_lat + P.Ts / 2 * F1_lat, y_lat)
        F3_lat = self.observer_f_lat(x_lat + P.Ts / 2 * F2_lat, y_lat)
        F4_lat = self.observer_f_lat(x_lat + P.Ts * F3_lat, y_lat)
        x_lat += P.Ts / 6 * (F1_lat + 2 * F2_lat + 2 * F3_lat + F4_lat)

        self.x_hat = np.array([
            [x_lat[0][0]],  
            [x_lon[0][0]],
            [x_lat[1][0]], 
            [x_lat[2][0]],
            [x_lon[1][0]], 
            [x_lat[3][0]]])

        return self.x_hat
    
    def observer_f_lon(self, x_hat, y_m):
        # xhatdot = A*xhat + B*u + L(y-C*xhat)
        xhat_dot = self.A_lon @ x_hat \
                   + self.B_lon * (self.F_d1-self.Fe) \
                   + self.L_lon @ (y_m-self.C_lon @ x_hat)
        return xhat_dot
    
    def observer_f_lat(self, x_hat, y_m):
        # xhatdot = A*xhat + B*u + L(y-C*xhat)
        xhat_dot = self.A_lat @ x_hat \
                   + self.B_lat * self.Tau_d1 \
                   + self.L_lat @ (y_m-self.C_lat @ x_hat)
        return xhat_dot


    def integratorAntiWindup(self, u_sat, u_unsat, ki, integrator):
        if ki != 0.0:
            integrator = integrator + P.Ts/ki*(u_sat-u_unsat)


def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u



