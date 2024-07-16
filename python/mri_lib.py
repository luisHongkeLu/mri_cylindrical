import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import scipy.interpolate
from scipy import special
logger = logging.getLogger(__name__)


def mri_evp(Re,Rm,r2,B0,O2,f_pchip,m,kz,Nr,Nphi,insulating = False):
    # Bases
    coords = d3.CartesianCoordinates('phi', 'r')
    dist = d3.Distributor(coords, dtype=np.complex128)
    rbasis = d3.ChebyshevT(coords['r'], size=Nr, bounds=(1, r2))
    phibasis = d3.ComplexFourier(coords['phi'], size=Nphi, bounds=(0, 2*np.pi))
    phi,r = dist.local_grids(phibasis, rbasis)
    annulus = (phibasis,rbasis)
    # Fields
    s = dist.Field(name='s')
    u = dist.Field(name='u', bases=annulus)
    v = dist.Field(name='v', bases=annulus)
    p = dist.Field(name='p', bases=annulus)
    br = dist.Field(name='br', bases=annulus)
    bphi = dist.Field(name='bphi', bases=annulus)
    tau_u1 = dist.Field(name='tau_u1', bases=phibasis)
    tau_v1 = dist.Field(name='tau_v1', bases=phibasis)
    tau_u2 = dist.Field(name='tau_u2', bases=phibasis)
    tau_v2 = dist.Field(name='tau_v2', bases=phibasis)
    tau_p1 = dist.Field(name='tau_p1', bases=phibasis)
    tau_p2 = dist.Field(name='tau_p2', bases=phibasis)
    tau_br1 = dist.Field(name='tau_br1', bases=phibasis)
    tau_br2 = dist.Field(name='tau_br2', bases=phibasis)
    tau_bphi1 = dist.Field(name='tau_bphi1', bases=phibasis)
    tau_bphi2 = dist.Field(name='tau_bphi2', bases=phibasis)

    ephi, er = coords.unit_vector_fields(dist)
    r_1 = dist.Field(name='r_1', bases=rbasis)
    r_1['g'] = r
    IdealTC = ( 1/r**2 - 1/r2**2 + O2*( 1 - 1/r**2 ))/(1 - 1/r2**2)  # \Omega(r) for ideal Couette flow
    Omega = dist.Field(name='Omega', bases=rbasis)
    # Omega['g'] = IdealTC
    Omega['g'] = f_pchip(r)
    bessel_i = special.iv(0, kz*1)/special.iv(1, kz*1)
    bessel_I = dist.Field(name='bessel_I', bases=rbasis)
    bessel_I = bessel_i
    bessel_k = special.kn(0, kz*r2)/special.kn(1, kz*r2)
    bessel_K = dist.Field(name='bessel_K', bases=rbasis)
    bessel_K = bessel_k
    
    # Substitutions
    Dt = lambda A: 1j*(-s +m*Omega)*A
    dr = lambda A: d3.Differentiate(A, coords['r'])
    dphi = lambda A: 1j*m*A
    dz = lambda A: 1j*kz*A
    lift_basis = rbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    lap_c = lambda A,tau: 1/r_1*dr(r_1*(dr(A)+lift(tau)))+(1/r_1**2)*dphi(dphi(A))+dz(dz(A))
    GS = lambda A, tau: dr(dr(A)+A/r_1+lift(tau))+(1/r_1**2)*dphi(dphi(A))+dz(dz(A))
    # grad_v = d3.grad(v) + er*lift(tau_v1)
    # grad_p = d3.grad(p) + er*lift(tau_p1)
    # Backgrounds
    # - 2*er*u0*ephi@u/r_1 +ephi*u0*er@u/r_1+ ephi*er@u*er@grad(u0)
    # Problem
    problem = d3.EVP([u, v, p,br,bphi,tau_br1,tau_br2,tau_bphi1,tau_bphi2, tau_u1, tau_v1, tau_u2, tau_v2, tau_p1,tau_p2], eigenvalue=s, namespace=locals())
    problem.add_equation("Dt(br) - B0*dz(u)- (1/Rm)*(GS(br,tau_br1)-(2/r_1**2)*dphi(bphi))+lift(tau_br2)=0")
    problem.add_equation("Dt(bphi)-B0*dz(v)-r_1*dr(Omega)*br-(1/Rm)*(GS(bphi,tau_bphi1)+(2/r_1**2)*dphi(br))+lift(tau_bphi2)=0")
    problem.add_equation("Dt(u) - 2*Omega*v+dr(p) -B0*dz(br)- (1/Re)*(GS(u,tau_u1)-(2/r_1**2)*dphi(v)) +lift(tau_u2) = 0")
    problem.add_equation("Dt(v) + 1/r_1*dphi(p) + (2*Omega+r_1*dr(Omega))*u - B0*dz(bphi)- (1/Re)*(GS(v,tau_v1)+(2/r_1**2)*dphi(u))+lift(tau_v2) = 0")
    problem.add_equation("2*(Omega+r_1*dr(Omega))*(1/r_1*dphi(u)-v/r_1)-2*Omega*dr(v)+(lap_c(p, tau_p1))+lift(tau_p2)=0")
    problem.add_equation("u(r=1) = 0")
    problem.add_equation("u(r=r2) = 0")
    problem.add_equation("v(r=1) = 0")
    problem.add_equation("v(r=r2) = 0")
    problem.add_equation("(er@grad(u)+u/r_1)(r=1)=0")
    problem.add_equation("(er@grad(u)+u/r_1)(r=r2)=0")
    if insulating:
        problem.add_equation("bphi(r=1) = 0")
        problem.add_equation("bphi(r=r2) = 0")
        problem.add_equation("(dr(r_1*br) - kz*r_1*bessel_I*br)(r=1) =0")
        problem.add_equation("(dr(r_1*br) + kz*r_1*bessel_K*br)(r=r2) =0")
    else:
        problem.add_equation("br(r=1) = 0")
        problem.add_equation("br(r=r2) = 0")
        problem.add_equation("(dr(r_1*bphi))(r=1) = 0")
        problem.add_equation("(dr(r_1*bphi))(r=r2) = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[m])
    return solver, u, v, p, br,bphi, dist,coords,annulus

def max_growth_rate(Re,Rm,r2,B0,O2,f_pchip,m,kz,Nr,Nphi,insulating):
    print(insulating)
    solver, u, v, p, br,bphi, dist,coords,annulus = mri_evp(Re,Rm,r2,B0,O2,f_pchip,m,kz,Nr,Nphi,insulating)
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    # evals2 = evals2[np.abs(evals2.imag)< 1e-3 ]
    return evals[np.argsort(-evals.imag)][0]
