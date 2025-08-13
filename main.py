# 1D, explicit, collocated code based on "On the extension of the AUSM+ scheme to compressible two-ﬂuid models Computers & Fluids 32 (2003) 891–916"
import numpy as np
import matplotlib.pyplot as plt
from functools import *

def P_plus(M): #Eq39
    if abs(M) >= 1:
        return M1_plus(M)/M
    else:
        return M2_plus(M)*(2 - M - 16*3/16*M*M2_minus(M))
    
def P_minus(M): #Eq39
    if abs(M) >= 1:
        return M1_minus(M)/M
    else:
        return  - M2_minus(M)*(2 + M - 16*3/16*M*M2_plus(M))

def M1_plus(M): #Eq36
    return 0.5*(M+abs(M))

def M1_minus(M): #Eq36
    return 0.5*(M-abs(M))

def M2_plus(M): #Eq37
    if abs(M) >= 1:
        return M1_plus(M)
    else:
        return 0.25*(M+1)**2
    
def M2_minus(M): #Eq37
    if abs(M) >= 1:
        return M1_minus(M)
    else:
        return -0.25*(M-1)**2

def M_plus(M): #Eq38
    if abs(M) >= 1:
        return M1_plus(M)
    else:
        return M2_plus(M)*(1-16*1/8*M2_minus(M))

def M_minus(M): #Eq38
    if abs(M) >= 1:
        return M1_minus(M)
    else:
        return M2_minus(M)*(1+16*1/8*M2_plus(M))

def mdotstar_k_alpha_kpstar(ak_L, ak_R, uk_L, uk_R, alphak_L, alphak_R, p_L, p_R, rhok_L, rhok_R):
    akstar = np.sqrt(ak_L*ak_R) # Eq32
    Mk_L = uk_L/akstar # Eq32
    Mk_R = uk_R/akstar # Eq32
    alphakpstar = P_plus(Mk_L)*alphak_L*p_L + P_minus(Mk_R)*alphak_R*p_R # Eq33
    Mkstar = M_plus(Mk_L) + M_minus(Mk_R) #Eq 34
    mdotkstar = akstar*(
                    alphak_L*rhok_L*
                        (Mkstar+abs(Mkstar))/2
                    +
                    alphak_R*rhok_R*
                        (Mkstar-abs(Mkstar))/2
    ) #Eq35
    return mdotkstar, alphakpstar

def F_k(ak_L, ak_R, uk_L, uk_R, alphak_L, alphak_R, p_L, p_R, rhok_L, rhok_R):
    mdotkstar, alphakpstar = mdotstar_k_alpha_kpstar(ak_L, ak_R, uk_L, uk_R, alphak_L, alphak_R, p_L, p_R, rhok_L, rhok_R)
    Psik_L = np.array([1, uk_L]) #Eq29
    Psik_R = np.array([1, uk_R])#Eq29
    F_ki = 0.5*mdotkstar*(Psik_L + Psik_R) + 0.5*abs(mdotkstar)*(Psik_L - Psik_R) + np.array([0, alphakpstar])#Eq29
    return F_ki[0], F_ki[1]

def p_dalphak_dx(p,alphak_plus1,alphak_minus1,deltax):
    return p * (alphak_plus1 - alphak_minus1)/(2*deltax) #End of Page 902

def delta_t(CFL,alpha,deltax,u_l,a_l,u_g,a_g): #Eq28
    return CFL* min(
                np.divide((1-alpha)*deltax,
                          (np.abs(u_l) + a_l))
                +
                np.divide((alpha)*deltax,
                          (np.abs(u_g) + a_g))
                
                )

def p_g_tait(rho_g, gamma=1.4, C=1E5, rho_g_0 = 1):
    return C*np.power(rho_g/rho_g_0, gamma) #Eq4

def p_l_tait(rho_l, B=3.3E8, n=7.15, rho_l_0 = 1000):
    return B*(np.power(rho_l/rho_l_0,n)-1) #Eq6

def rho_g_tait(p, gamma=1.4, C=1E5, rho_g_0 = 1):
    return np.power(p/C,1/gamma)*rho_g_0 #Eq4

def rho_l_tait(p, B=3.3E8, n=7.15, rho_l_0 = 1000):
    return np.power(p/B + 1, 1/n)*rho_l_0 #Eq6

def a_g_tait(rho_g, gamma=1.4, C=1E5, rho_g_0 = 1): #Eq5
    p = p_g_tait(rho_g, gamma=gamma, C=C, rho_g_0 = rho_g_0)
    return np.sqrt(gamma*p/rho_g) #Eq5

def a_l_tait(rho_l, B=3.3E8, n=7.15, rho_l_0 = 1000):
    p = p_l_tait(rho_l, B=B,n=n,rho_l_0=rho_l_0)#Eq 6
    return np.sqrt(n/rho_l * (p+B)) #Eq7

def cons_variables(alpha, rho_g, rho_l, u_g, u_l): 
    U = np.column_stack((alpha*rho_g, (1-alpha)*rho_l, alpha*rho_g*u_g, (1-alpha)*rho_l*u_l)) #Eq3
    return U

def primitive_variables(U):
    u_g = np.divide(U[:,2],U[:,0])
    u_l = np.divide(U[:,3],U[:,1])
    p = np.empty(len(U))
    for cell_i in range(len(U)):
        p[cell_i] = solve_p(U[cell_i,0],U[cell_i,1]) #Newton-raphson method, Eq12
    alpha = np.divide(U[:,0],rho_g_tait(p))
    return alpha, u_g, u_l, p

def solve_p(U1,U2):
    p = 1E5
    while abs(F_p(p,U1,U2)) > 1E-12:
        p = p - (F_p(p,U1,U2) / Fprime_p(p,U1,U2)) #Newton-raphson method 
    return p
    
def F_p(p, U1,U2):
    gamma = 1.4
    C=1E5
    rho_g_0 = 1
    rho_l_0 = 1000
    B = 3.3E8
    n = 7.15
    return (
        (1 - U1/
         ((p/C)**(1/gamma) * rho_g_0))
         *
         ((p/B+1)**(1/n) * rho_l_0) - U2
    ) #Eq12

def Fprime_p(p, U1, U2):
    gamma = 1.4
    C=1E5
    rho_g_0 = 1
    rho_l_0 = 1000
    B = 3.3E8
    n = 7.15
    return (
        (U1*rho_l_0*(p/B + 1)**(1/n) * (p/C)**(-1/gamma-1)) / (C*gamma*rho_g_0)
    +   (rho_l_0*(p/B + 1)**(1/n-1) * (1 - (U1*(p/C)**(-1/gamma)) / (rho_g_0))) / (B*n)
    ) #Derivative of Eq12 WRT p

def analytical_solution(x,t):
    alpha = 0.2*np.ones(len(x))
    for i, xi in enumerate(x):
        if xi < 9.81*t**2/2 + 10*t:
            alpha[i] = 1-(1-0.2)*10/(np.sqrt(10**2+2*9.81*xi)) #Eq46
    return alpha 

def F_k_nv(alphak_plus1, alphak_minus1, deltax, p, sigma, alpha_g, rho_g, alpha_l, rho_l, u_g, u_l):
    dalphak_dx = (alphak_plus1-alphak_minus1)/(2*deltax) #Eq23
    pint = p - sigma*(alpha_g * rho_g * alpha_l * rho_l)/(alpha_g * rho_l + alpha_l * rho_g) * (u_g - u_l)**2
    return (pint - p)*dalphak_dx #24

# Build mesh (+2 is because we use ghost cells to enforce BCs)
cells = 100
x = np.linspace(0,12.5,cells+2)
dx = x[1]-x[0]

rho_g = np.empty(cells+2)
rho_l = np.empty(cells+2)

u_l = np.empty(cells+2)
u_g = np.empty(cells+2)

a_l = np.empty(cells+2)
a_g = np.empty(cells+2)

alpha = np.empty(cells+2)
p = np.empty(cells+2)

Fstar_plus = np.zeros((cells+2,4))
Fstar_minus = np.zeros((cells+2,4))
C_nv = np.zeros((cells+2,4))

# Time discretization
t_max = 0.5 # End time for simulation
CFL = 0.5 # We run at a fixed maximum CFL number
t = 0 # Current time

# Initial conditions
alpha[:] = 0.2
u_l[:] = 10
u_g[:] = 0
p[:] = 1E5


while t < t_max:

    # Update useful properties
    rho_g = rho_g_tait(p)
    rho_l = rho_l_tait(p)
    a_g = a_g_tait(rho_g)
    a_l = a_l_tait(rho_l)

    # Eq3, map from flow variables (primitive variables) to conserved variables U
    U = cons_variables(alpha = alpha,
                       rho_g = rho_g,
                       rho_l = rho_l,
                       u_g = u_g,
                       u_l = u_l)
    
    for cell_i in range(1,cells+1):    
        #Fstar(j-1, j) is indexed by cell j 
        #Fstar(j, j+1) is indexed by cell j+1 
        # Cells:       .   |   .   |   .   |
        # Cell index:          j
        # Fstar index:     j      j+1
        Fstar_plus[cell_i,0],Fstar_plus[cell_i,2] = F_k(ak_L = a_g[cell_i],
                                                    ak_R = a_g[cell_i+1],
                                                    uk_L = u_g[cell_i],
                                                    uk_R = u_g[cell_i+1],
                                                    alphak_L = alpha[cell_i],
                                                    alphak_R = alpha[cell_i+1],
                                                    p_L = p[cell_i],
                                                    p_R = p[cell_i+1],
                                                    rhok_L = rho_g[cell_i],
                                                    rhok_R = rho_g[cell_i+1],
                                                    ) #Eq27, Fstar_plus returns mass and momentum fstar for gas
        
        Fstar_plus[cell_i,1],Fstar_plus[cell_i,3] = F_k(ak_L = a_l[cell_i],
                                                    ak_R = a_l[cell_i+1],
                                                    uk_L = u_l[cell_i],
                                                    uk_R = u_l[cell_i+1],
                                                    alphak_L = 1-alpha[cell_i],
                                                    alphak_R = 1-alpha[cell_i+1],
                                                    p_L = p[cell_i],
                                                    p_R = p[cell_i+1],
                                                    rhok_L = rho_l[cell_i],
                                                    rhok_R = rho_l[cell_i+1],
                                                    ) #Eq27, Fstar_plus returns mass and momentum fstar for liquid
        
        Fstar_minus[cell_i,0],Fstar_minus[cell_i,2] = F_k(ak_L = a_g[cell_i-1],
                                                    ak_R = a_g[cell_i],
                                                    uk_L = u_g[cell_i-1],
                                                    uk_R = u_g[cell_i],
                                                    alphak_L = alpha[cell_i-1],
                                                    alphak_R = alpha[cell_i],
                                                    p_L = p[cell_i-1],
                                                    p_R = p[cell_i],
                                                    rhok_L = rho_g[cell_i-1],
                                                    rhok_R = rho_g[cell_i],
                                                    ) #Eq27, Fstar_minus returns mass and momentum fstar for liquid
        
        Fstar_minus[cell_i,1],Fstar_minus[cell_i,3] = F_k(ak_L = a_l[cell_i-1],
                                                    ak_R = a_l[cell_i],
                                                    uk_L = u_l[cell_i-1],
                                                    uk_R = u_l[cell_i],
                                                    alphak_L = 1-alpha[cell_i-1],
                                                    alphak_R = 1-alpha[cell_i],
                                                    p_L = p[cell_i-1],
                                                    p_R = p[cell_i],
                                                    rhok_L = rho_l[cell_i-1],
                                                    rhok_R = rho_l[cell_i],
                                                    ) #Eq27, Fstar_minus returns mass and momentum fstar for liquid
        
        # C_nv is non-viscous terms, Eq3
        # Note that the gravity term alpha*rho*g is added into C_nv, when it should technically be in something called S
        # This is OK since C_nv and S are added in Eq27 anyways
        C_nv[cell_i,2] = p_dalphak_dx(p[cell_i],
                                      alphak_plus1 = alpha[cell_i+1],
                                      alphak_minus1 = alpha[cell_i-1],
                                      deltax=dx) + alpha[cell_i]*rho_g[cell_i]*9.81 \
                                    + F_k_nv(alphak_plus1 = alpha[cell_i+1],
                                             alphak_minus1 = alpha[cell_i-1],
                                             deltax=dx,
                                             p = p[cell_i],
                                             sigma=2, # Needs to be parameterized rather than hard coded down here
                                             alpha_g = alpha[cell_i],
                                             rho_g = rho_g[cell_i],
                                             alpha_l = 1-alpha[cell_i],
                                             rho_l = rho_l[cell_i],
                                             u_g = u_g[cell_i],
                                             u_l = u_l[cell_i])
        
        C_nv[cell_i,3] = p_dalphak_dx(p[cell_i],
                                      alphak_plus1 = 1-alpha[cell_i+1],
                                      alphak_minus1 = 1-alpha[cell_i-1],
                                      deltax=dx)+ (1-alpha[cell_i])*rho_l[cell_i]*9.81 \
                                    + F_k_nv(alphak_plus1 = 1 - alpha[cell_i+1],
                                             alphak_minus1 = 1- alpha[cell_i-1],
                                             deltax=dx,
                                             p = p[cell_i],
                                             sigma=2, # Needs to be parameterized rather than hard coded down here
                                             alpha_g = alpha[cell_i],
                                             rho_g = rho_g[cell_i],
                                             alpha_l = 1-alpha[cell_i],
                                             rho_l = rho_l[cell_i],
                                             u_g = u_g[cell_i],
                                             u_l = u_l[cell_i])
    

    # Compute dt based on CFL number (Eq28)
    dt = delta_t(CFL,alpha,dx,u_l,a_l,u_g,a_g)

    # Main flow variable update
    U = U - dt/dx*(Fstar_plus - Fstar_minus) + dt*C_nv #Eq27
    t = t+dt
    # Eq9-12, map conserved variables back to primitive variables 
    alpha, u_g, u_l, p = primitive_variables(U)

    # Boundary conditions
    # Enforced on the primitive variables (they will be mapped back to conservative variables at the start of the next time step)

    # Inlet - fixed u, alpha, zero-gradient p
    u_g[0] = 0
    u_l[0] = 10
    alpha[0] = .2
    p[0] = p[1]

    # Outlet - fixed p, zero-gradient u, alpha
    u_g[-1] = u_g[-2]
    u_l[-1] = u_l[-2]
    alpha[-1] = alpha[-2]
    p[-1] = 1E5

    """
    # These BCs are for the phase separation problem (walls at top and bottom)
    u_g[0] = -u_g[1]
    u_l[0] = -u_l[1]
    alpha[0] = alpha[1]
    p[0] = p[1]
    rho_g[0] = rho_g[1]
    rho_l[0] = rho_l[1]

    u_g[-1] = -u_g[-2]
    u_l[-1] = -u_l[-2]
    alpha[-1] = alpha[-2]
    p[-1] = p[-2]
    rho_g[-1] = rho_g[-2]
    rho_l[-1] = rho_l[-2]
    """
    
    print(f'Time: {t}')
    print(f'    dt: {dt}')

    if t > t_max:
        break

alpha_ana = analytical_solution(x, t)
plt.scatter(x,alpha,label='Simulation')
plt.plot(x,alpha_ana,label='Analytical solution')
plt.legend()
plt.savefig('alpha.png')
print('done')





