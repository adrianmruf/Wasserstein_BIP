import numpy as np

import matplotlib.pyplot as plt

import scipy.integrate as integrate


class FVM:
    # open boundary conditions

    # not really happy with the solution of having
    # arrays for u as class attrbutes instead of
    # function arguments/returns. It makes stuff
    # pretty unclear and the performance gain
    # is dubious and hypothetical anyway.

    def __init__(self, domain, N):
        # self.IC = IC
        dx = (domain[1] - domain[0]) / N
        # include ghost-cells
        self.N = N + 2

        self.a = domain[0]
        self.b = domain[1]

        # self.x are the cell centers
        self.x, self.dx = np.linspace(start=domain[0] - .5 * dx,
                                      stop=domain[1] + .5 * dx,
                                      num=self.N,
                                      retstep=True)

        self.u = np.zeros_like(self.x)
        # self.u_star = np.zeros_like(self.u)
        # self.dudt = np.zeros_like(self.u)

    def integrate(self, IC, fluxL, fluxR, fluxRInverse, cfl, T, only_endstate=True):
        # self.u[:] = np.array([self.IC(x_) for x_ in self.x], dtype=np.float)
        # for i in range(self.N):
        #      # initialize u_0 with cell averages
        #      x = self.x[i]
        #      domain = np.linspace(x - .5 * self.dx, x + .5 * self.dx)

        #      self.u[i] = np.trapz(y=[self.IC(x_) for x_ in domain], x=domain) / self.dx
             

        M = int(T*(self.N-2)/(cfl*(self.b-self.a)))+1
        dt = float(T)/M
        N_comp = self.N-2
        N_comp_half = int(N_comp/2)
        self.u_new = np.zeros((M+1,N_comp))

        for j in range(0,N_comp,1):
            self.u_new[0,j] = (1/self.dx)*integrate.quad(IC,self.a+j*self.dx,self.a+(j+1)*self.dx)[0]

        for n in range(0,M,1):
            # self._step(dt, fluxL, fluxR, fluxRInverse)
            for i in range(0,N_comp_half,1):
                self.u_new[n+1,i] = self.u_new[n,i] -(dt/self.dx)*(fluxL(self.u_new[n,i])-fluxL(self.u_new[n,self.mod(i-1)]))

            self.u_new[n+1,N_comp_half] = fluxRInverse(fluxL(self.u_new[n+1,N_comp_half-1]))

            for i in range(N_comp_half+1,N_comp,1):
                self.u_new[n+1,i] = self.u_new[n,i] -(dt/self.dx)*(fluxR(self.u_new[n,i])-fluxR(self.u_new[n,i-1]))
        
    def mod(self,i):
        if i<0:
            return i+1
        else:
            return i

        # t = 0

        # if not only_endstate:
        #     U = [np.copy(self.u)]  # store sequence of states
        #     Ts = [t]               # store sequence of timepoints

        # while (t < T):
        #     dt = cfl*self.dx
        #     t += dt

        #     # writes new state into self.u
        #     self._step(dt, fluxL, fluxR, fluxRInverse)

        #     if not only_endstate:
        #         U += [np.copy(self.u)]
        #         Ts += [t]

        # if not only_endstate:
        #     U_ = np.empty((len(self.u), len(Ts)))
        #     for i in range(len(Ts)):
        #         U_[:, i] = U[i]

        #     T_ = np.array(Ts)

        #     return U_[1:-1, :], T_

        # return np.copy(self.u[1:-1]), t

    def _step(self, dt, fluxL, fluxR, fluxRInverse):
        """Write state after dt evolution of self.u into self.u"""
        
        N_comp = self.N-2
        N_comp_half = int(N_comp/2)
        u = np.copy(self.u)
        for i in range(1,N_comp_half+1,1):
            u[i] = u[i] -(dt/self.dx)*(fluxL(u[i])-fluxL(u[i-1]))

        u[N_comp_half+1] = fluxRInverse(fluxL(u[N_comp_half]))

        for i in range(N_comp_half+2,N_comp+1,1):
            u[i] = u[i] -(dt/self.dx)*(fluxR(u[i])-fluxR(u[i-1]))

        self._apply_bc(u)
        self.u = u

        # # SSPRK2
        # self._rate_of_change(self.u, dt, flux, flux_prime)
        # self.u_star = self.u + dt * self.dudt
        # self._apply_bc(self.u_star)

        # self._rate_of_change(self.u_star, dt, flux, flux_prime)
        # self.u_star += dt * self.dudt

        # self.u += self.u_star
        # self.u /= 2
        # self._apply_bc(self.u)

    # def _rate_of_change(self, u_0, dt, flux, flux_prime):
    #     """Write 1 / dx (F_j+1/2 - F_j-1/2) based on u_0 into dudt"""
    #     self._flux_difference(u_0, flux, flux_prime)
    #     self.dudt /= -self.dx

    # def _flux_difference(self, u_0, flux, flux_prime):
    #     """write (F_j+1/2 - F_j-1/2) based on u_0 into dudt"""
    #     # flux over left boundary, to be updated in loop
    #     f_right = self._flux(u_0[0], u_0[1], flux, flux_prime)

    #     for i in range(1, self.N - 1):
    #         f_left = f_right
    #         f_right = self._flux(u_0[i], u_0[i+1], flux, flux_prime)

    #         self.dudt[i] = f_right - f_left

    # def _flux(self, u_l, u_r, f, fp):
    #     flux_average = 0.5 * (f(u_l) + f(u_r))
    #     speed = max(abs(fp(u_l)), abs(fp(u_r)))

    #     return flux_average - 0.5 * speed * (u_r - u_l)

    def _apply_bc(self, u):
        ### periodic
        # u[0]=u[-2]
        # u[-1]=u[1]
        ### outflow
        u[0] = u[1]
        u[-1] = u[-2]




def TransportFlux(u, delta):
    return (1+delta)*u

def TransportFluxInverse(u, delta):
    return u/(1+delta)
    
def BurgersFlux(u, delta):
    return 0.5*u**2

def BurgersFluxInverse(u, delta):
    return np.sqrt(2*u)




def show_Buckley_Leverett_riemann():
    for i in np.arange(-0.2, 0.3, 0.1):
        for j in np.arange(-0.3, 0.4, 0.1):
            r = FVM((-1.,1.),
                    128)
            r.integrate(IC=lambda x: 0.5+i + (1.5-i)*float(x>-0.5),
                        fluxL=lambda u: TransportFlux(u,j),
                        fluxR=lambda u: BurgersFlux(u,0),
                        fluxRInverse=lambda u: BurgersFluxInverse(u,0),
                        cfl=0.4,
                        T=1.,
                        only_endstate=False)
            U = r.u_new
            plt.plot(r.x[1:-1],U[-1,:])

    # plt.show()
    r = FVM((-1.,1.),
                    128)
    r.integrate(IC=lambda x: 0.5+0.1 + (1.5-0.1)*float(x>-0.5),
                fluxL=lambda u: TransportFlux(u,0.1),
                fluxR=lambda u: BurgersFlux(u,0.),
                fluxRInverse=lambda u: BurgersFluxInverse(u,0),
                cfl=0.4,
                T=1.,
                only_endstate=False)
    np.savetxt('Discontinuous_flux_prior_mean.txt',r.u_new[-1,:])
    # np.savetxt('x.txt',r.x[1:-1])


if __name__ == '__main__':
    # test_RunanovFVM()
    show_Buckley_Leverett_riemann()
