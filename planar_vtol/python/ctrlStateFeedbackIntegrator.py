import numpy as np
import VTOLParam as P
import control as cnt

class ctrlStateFeedbackIntegrator:
    def __init__(self):
        # initialize wind force (this term was always in the dynamics solution, but may not be
        # included in your own, so please check).
        P.F_wind = 0.1

        # tuning parameters 
        wn_h = 1.0
        zeta_h = 0.95
        wn_z = 0.9905
        zeta_z = 0.95
        wn_th = 13.3803
        zeta_th = 0.95
        integrator_h = 1.0
        integrator_z = 1.0

        # State Space Equations
        self.Fe = (P.mc + 2.0 * P.mr) * P.g  # equilibrium force 
        A_lon = np.array([[0.0, 1.0],
                        [0.0, 0.0]])
        B_lon = np.array([[0.0],
                        [1.0 / (P.mc + 2.0 * P.mr)]])
        C_lon = np.array([[1.0, 0.0]])
        A_lat = np.array([[0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, -self.Fe / (P.mc + 2.0 * P.mr), -(P.mu / (P.mc + 2.0 * P.mr)), 0.0],
                        [0.0, 0.0, 0.0, 0.0]])
        B_lat = np.array([[0.0],
                        [0.0],
                        [0.0],
                        [1.0 / (P.Jc + 2 * P.mr * P.d ** 2)]])
        C_lat = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]])
        
        # form augmented system
        A1_lon = np.vstack((
                np.hstack((A_lon, np.zeros((2,1)))),
                np.hstack((-C_lon, np.zeros((1,1))))))
        B1_lon = np.vstack((B_lon, np.zeros((1,1))))
        A1_lat = np.vstack((
                np.hstack((A_lat, np.zeros((4,1)))),
                np.hstack((-C_lat[0:1], np.zeros((1,1))))))
        B1_lat = np.vstack((B_lat, np.zeros((1,1))))

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
        self.F_limit = P.max_thrust * 2.0
        self.tau_limit = P.max_thrust * P.d * 2.0

    def update(self, r, x):
        z_r = r[0][0]
        h_r = r[1][0]
        z = x[0][0]
        h = x[1][0]
        theta = x[2][0]

        # integrate error
        error_z = z_r - z
        self.integrator_z += (P.Ts/2.0)*(error_z + self.error_z_d1)
        self.error_z_d1 = error_z

        error_h = h_r - h
        self.integrator_h += (P.Ts/2.0)*(error_h + self.error_h_d1)
        self.error_h_d1 = error_h

        # Construct the states
        x_lon = np.array([[x[1][0]], [x[4][0]]])
        x_lat = np.array([[x[0][0]], [x[2][0]], [x[3][0]], [x[5][0]]])

        # Compute the state feedback controllers
        F_tilde = -self.K_lon @ x_lon - self.ki_lon * self.integrator_h
        F = self.Fe / np.cos(theta) + F_tilde[0]
        F_sat = saturate(F, self.F_limit)
        self.integratorAntiWindup(F_sat, F, self.ki_lon, self.integrator_h)

        tau = -self.K_lat @ x_lat - self.ki_lat*self.integrator_z
        tau_sat = saturate(tau[0], self.tau_limit)
        self.integratorAntiWindup(tau_sat, tau, self.ki_lat, self.integrator_z)

        return np.array([[F_sat], [tau_sat]])

    def integratorAntiWindup(self, u_sat, u_unsat, ki, integrator):
        if ki != 0.0:
            integrator = integrator + P.Ts/ki*(u_sat-u_unsat)


def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u



