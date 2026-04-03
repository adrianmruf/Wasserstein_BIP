
import numpy as np

import matplotlib.pyplot as plt


class EulersEquation:
    # w is conservative variables:
    # rho, m, E
    # density, momentum (rho*v), Energy
    dimension = 3
    gamma = 1.4

    @classmethod
    def flux(cls, w):
        rho, v, p = cls.conservative_to_primitive(w)
        m = w[1]
        E = w[2]
        return np.array([m,
                         rho*v*v + p,
                         (E + p) * v])

    @classmethod
    def conservative_to_primitive(cls, w):
        # primitive: rho, v, p
        return np.array([w[0],
                         w[1] / w[0],
                         cls.pressure(w)])

    @classmethod
    def primitive_to_conservative(cls, w_primitive):
        rho = w_primitive[0]
        v = w_primitive[1]
        p = w_primitive[2]
        return np.array([rho,
                         rho * v,
                         p / (cls.gamma - 1) + 0.5 * rho * v * v])

    @classmethod
    def sound_speed(cls, w):
        return np.sqrt(cls.gamma * cls.pressure(w) / w[0])

    @classmethod
    def pressure(cls, w):
        rho = w[0]
        v = w[1] / w[0]
        E = w[2]
        return (cls.gamma - 1) * (E - 0.5*rho*v*v)

    @classmethod
    def eigenvalues(cls, w):
        v = w[1] / w[0]
        c = cls.sound_speed(w)
        return np.array([v - c, v, v + c])

    @classmethod
    def max_eigenvalue(cls, w):
        return w[1] / w[0] + cls.sound_speed(w)


class EulerHLLC:
    # works in conservative variables,
    # IC and output in primitive variables
    def __init__(self, model, domain, N):
        # model: handles equation-dependent stuff,
        # fulfills the same requirements as EulersEquation
        self.M = model.dimension
        self.model = model
        dx = (domain[1] - domain[0]) / N
        # include ghost-cells
        self.N = N + 2
        # self.x are the cell centers
        self.x, self.dx = np.linspace(start=domain[0] - .5 * dx,
                                      stop=domain[1] + .5 * dx,
                                      num=self.N,
                                      retstep=True)

        self.u = np.zeros((self.N, self.M))
        self.u_star = np.zeros_like(self.u)
        self.dudt = np.zeros_like(self.u)

    def integrate(self, IC, T, only_endstate=True):
        for i in range(self.N):
            # initialize u_0 with cell averages
            x = self.x[i]
            cell = np.linspace(x - .5 * self.dx, x + .5 * self.dx)
            self.u[i, :] = np.trapz(y=[IC(x_) for x_ in cell], x=cell, axis=0) / self.dx
            self.u[i, :] = self.model.primitive_to_conservative(self.u[i, :])

        t = 0

        if not only_endstate:
            U = [np.copy(self.u)]  # store sequence of states
            Ts = [t]               # store sequence of timepoints

        while (t < T):
            dt = self._cfl()
            t += dt

            # writes new state into self.u
            self._step(dt)

            if not only_endstate:
                # convert from conservative to primitive
                u_temp = np.copy(self.u)
                for i in range(self.N):
                    u_temp[i, :] = self.model.conservative_to_primitive(u_temp[i, :])
                U += [u_temp]
                Ts += [t]

        if not only_endstate:
            U_ = np.empty((self.N, self.M, len(Ts)))
            for i in range(len(Ts)):
                U_[:, i] = U[i]

            T_ = np.array(Ts)

            return U_[1:-1, :], T_

        # convert from conservative to primitive
        u_temp = np.copy(self.u)
        for i in range(self.N):
            u_temp[i, :] = self.model.conservative_to_primitive(u_temp[i, :])
        return u_temp[1:-1, :], t

    def _step(self, dt):
        """Write state after dt evolution of self.u into self.u"""
        # SSPRK2
        self._rate_of_change(self.u, dt)
        self.u_star = self.u + dt * self.dudt
        self._apply_bc(self.u_star)

        self._rate_of_change(self.u_star, dt)
        self.u_star += dt * self.dudt

        self.u += self.u_star
        self.u /= 2
        self._apply_bc(self.u)

    def _rate_of_change(self, u_0, dt):
        """Write 1 / dx (F_j+1/2 - F_j-1/2) based on u_0 into dudt"""
        self._flux_difference(u_0)
        self.dudt /= -self.dx

    def _flux_difference(self, u_0):
        """write (F_j+1/2 - F_j-1/2) based on u_0 into dudt"""
        # flux over left boundary, to be updated in loop
        f_right = self._flux(u_0[0, :], u_0[1, :])

        for i in range(1, self.N - 1):
            f_left = f_right
            f_right = self._flux(u_0[i, :], u_0[i+1, :])

            self.dudt[i, :] = f_right - f_left

    def _flux(self, uL, uR):
        fL = self.model.flux(uL)
        # eigenvalues are ordered
        sL = self.model.eigenvalues(uL)[0]
        rhoL, vL, pL = self.model.conservative_to_primitive(uL)

        fR = self.model.flux(uR)
        # eigenvalues are ordered
        sR = self.model.eigenvalues(uR)[-1]
        rhoR, vR, pR = self.model.conservative_to_primitive(uR)

        # Roe pressure
        sM = ((pR - pL + rhoL * vL * (sL - vL) - rhoR * vR * (sR - vR))
              / (rhoL * (sL - vL) - rhoR * (sR - vR)))
        pM = 0.5 * (pR + pL + rhoL * (sL - vL) * (sM - vL) + rhoR * (sM - vR) * (sR - vR))

        # numerical flux depending on which regime is at the boundary
        if sL >= 0:
            return fL
        elif sL <= 0 and sM > 0:
            rhoLs = rhoL * (vL - sL) / (sM - sL)
            uLs = self.model.primitive_to_conservative([rhoLs, sM, pM])
            return fL + sL * (uLs - uL)
        elif sM <= 0 and sR >= 0:
            rhoRs = rhoR * (vR - sR) / (sM - sR)
            uRs = self.model.primitive_to_conservative([rhoRs, sM, pM])
            return fR + sR * (uRs - uR)
        else:  # sR <= 0
            return fR

    def _apply_bc(self, u):
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]

    def _cfl(self):
        """dt = 0.5 * dx / max(|f'(u)|)"""
        max_speed = 0
        # this could probably be vectorized
        for i in range(1, self.N - 1):
            max_speed = max(max_speed, abs(self.model.max_eigenvalue(self.u[i])))

        return 0.5 * self.dx / max_speed


def show_sod_shock_tube():
    def sod_IC(x):
        u = np.zeros((3,))
        if x <= 0.5:
            u[0] = 1
            u[1] = 0
            u[2] = 1
        else:
            u[0] = 0.125
            u[1] = 0
            u[2] = 0.1
        return u

    r = EulerHLLC(EulersEquation, (0,1), 128)

    U, T = r.integrate(sod_IC, 0.2, True)

    plt.plot(r.x[1:-1], U[:, 0])
    plt.title("Density, T=0.2")
    plt.show()
    plt.plot(r.x[1:-1], U[:, 1], label="velocity")
    plt.title("Velocity, T=0.2")
    plt.show()
    plt.plot(r.x[1:-1], U[:, 2], label="pressure")
    plt.title("Pressure, T=0.2")
    plt.show()


if __name__ == '__main__':
    show_sod_shock_tube()