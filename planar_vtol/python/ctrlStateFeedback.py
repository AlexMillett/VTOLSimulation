import numpy as np
import VTOLParam as P
import control as cnt

class ctrlStateFeedback:
    def __init__(self):
        # tuning parameters
        tr_h = 2.0
        wn_h = 2.2 / tr_h
        zeta_h = 0.707
        tr_z = 2.0
        wn_z = 2.2 / tr_z
        zeta_z = 0.707
        M = 10.0
        tr_th = tr_z / M
        wn_th = 2.2 / tr_th
        zeta_th = 0.707
        # It is important to note that we formulate the dynamics as longitudinal dynamics
        # and lateral dynamics, but this is not strictly necessary. It may help us to see
        # and understand the effect of the two parts that are mostly decoupled (for the
        # linearized system at least). However, it is totally acceptable to also form
        # one large set of state space equations that includes A_lon and A_lat in one
        # A matrix (similary for B, C, etc.).
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
        # gain calculation
        des_char_poly_lon = [1.0, 2.0 * zeta_h * wn_h, wn_h ** 2]
        des_poles_lon = np.roots(des_char_poly_lon)
        des_char_poly_lat = np.convolve([1.0, 2.0 * zeta_z * wn_z, wn_z ** 2],
                                        [1.0, 2.0 * zeta_th * wn_th, wn_th ** 2])
        des_poles_lat = np.roots(des_char_poly_lat)
        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A_lon, B_lon)) != 2:
            print("The longitudinal system is not controllable")
        else:
            self.K_lon = cnt.acker(A_lon, B_lon, des_poles_lon)
            self.kr_lon = -1.0 / (C_lon @ np.linalg.inv(A_lon - B_lon @ self.K_lon) @ B_lon)
        if np.linalg.matrix_rank(cnt.ctrb(A_lat, B_lat)) != 4:
            print("The lateral system is not controllable")
        else:
            self.K_lat = cnt.acker(A_lat, B_lat, des_poles_lat)
            Cr_lat = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr_lat = -1.0 / (Cr_lat @ np.linalg.inv(A_lat - B_lat @ self.K_lat) @ B_lat)
        print('K_lon: ', self.K_lon)
        print('kr_lon: ', self.kr_lon)
        print('K_lat: ', self.K_lat)
        print('kr_lat: ', self.kr_lat)

    def update(self, r, x):
        z_r = r[0][0]
        h_r = r[1][0]
        z = x[0][0]
        h = x[1][0]
        theta = x[2][0]
        # Construct the states
        x_lon = np.array([[x[1,0]], [x[4,0]]])
        x_lat = np.array([[x[0,0]], [x[2,0]], [x[3,0]], [x[5,0]]])
        # Compute the state feedback controllers
        F_tilde = -self.K_lon @ x_lon + self.kr_lon * h_r
        F = self.Fe / np.cos(theta) + F_tilde[0][0]
        tau = -self.K_lat @ x_lat + self.kr_lat * z_r
        return np.array([[F], [tau[0][0]]])


def saturate(u, limit):
    if abs(u) > limit:
        u = limit*np.sign(u)
    return u

