import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import scipy.interpolate
from scipy import special
logger = logging.getLogger(__name__)


def find_common_elements_with_tolerance(arr1, arr2, tol):
    common_elements = []
    for elem in arr1:
        if np.any(np.isclose(elem, arr2, atol=tol)):
            common_elements.append(elem)
    return np.array(common_elements)

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

def main():
    data = np.genfromtxt('/Users/luis_lu/Desktop/PPPL/baseflow/Ideal_Couette_Omega2=0.19.txt')
    #data = np.genfromtxt('data.txt')
    r_data = data[:,0]
    Omega_data = data[:,1]
    f_pchip = scipy.interpolate.PchipInterpolator(r_data,Omega_data)

    # Parameters
    Rm =100
    Re = 1e5
    r2 = 3
    B0 = 0.12
    n = 2
    h = 4
    m_max = 300
    m = 1
    kz = n*np.pi/h
    Nphi = 2 * m_max + 2
    Nr = 196
    O2 = 0.19  # Omega_out/Omega_in
    insulating = True

    solver1, u1, v1, p1, br1,bphi1, dist1,coords1,annulus1 = mri_evp(Re,Rm,r2,B0,O2,f_pchip,m,kz,Nr,Nphi,insulating)
    evals1 = solver1.eigenvalues[np.isfinite(solver1.eigenvalues)]
    solver2, u2, v2, p2, br2,bphi2, dist2,coords2,annulus2 = mri_evp(Re,Rm,r2,B0,O2,f_pchip,m,kz,Nr,Nphi,insulating)
    evals2 = solver2.eigenvalues[np.isfinite(solver2.eigenvalues)]


    # Plot the complex numbers on the complex plane
    plt.figure(figsize=(8, 6))
    eigenvalue_spectrum = find_common_elements_with_tolerance(evals1, evals2, 1e-3)
    plt.scatter(eigenvalue_spectrum.real, eigenvalue_spectrum.imag, color='green', marker='o')

    # plt.scatter(evals1.real, evals1.imag, color='green', marker='o')
    # plt.scatter(evals2.real, evals2.imag, color='red', marker='o',alpha=0.5)
    # eigenvalue_spectrum = eigenvalue_spectrum[np.abs(eigenvalue_spectrum.imag)< 1e-5 ]
    max_eigenvalue = eigenvalue_spectrum[np.argsort(-eigenvalue_spectrum.imag)][0]
    # max_eigenvalue = evals2[np.argsort(-evals2.real)][0]
    plt.scatter(max_eigenvalue.real,max_eigenvalue.imag,color = "blue", label = f"{max_eigenvalue.real},{max_eigenvalue.imag}i")
    # plt.title('Eigenvalues')
    plt.legend()
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    # plt.ylim(-10, 10)
    # plt.xlim(-1, 1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"/Users/luis_lu/Desktop/PPPL/mri_cylindrical/plots/MRI_spectrum_{insulating}_1e{np.log10(Re)}_Rm={Rm}_m={m}_n={n}_Nr={Nr}.png",dpi=300)
    plt.clf()
    #EigenFunc_1e',num2str(log10(Re)),'_',base_flow,'_Rm=',num2str(Rm),'_B0=',num2str(B0),'_m=',num2str(m),'_n=',num2str(n),'_Nr=',num2str(Nr),'.txt'

    bz = dist2.Field(name='bz', bases=annulus2)
    w = dist2.Field(name='w', bases=annulus2)
    phi,r = dist2.local_grids(annulus2[0], annulus2[1])
    r_1 = dist2.Field(name='r_1', bases=annulus2[1])
    dr = lambda A: d3.Differentiate(A, coords2['r'])
    r_1['g'] = r
    bz = -(1/r_1*(dr(r_1*br2))+1/r_1*1j*m*bphi2)/(1j*kz)
    w = -(1/r_1*(dr(r_1*u2))+1/r_1*1j*m*v2)/(1j*kz)

    r = dist2.local_grid(annulus2[1])
    solver2.set_state(np.argmin(np.abs(solver2.eigenvalues - max_eigenvalue)), 0)
    r = dist2.local_grid(annulus2[1])
    phi = dist2.local_grid(annulus2[0])
    cmap = 'RdBu_r'
    plt.title("eigenfunctions")
    plt.plot(r[0,:],np.abs(u2['g'][0,:]),label = r"$u_r$")
    plt.plot(r[0,:],np.abs(v2['g'][0,:]),label = r"$u_\phi$")
    plt.plot(r[0,:],np.abs(w['g'][0,:]),label = r"$u_z$")
    plt.plot(r[0,:],np.abs(br2['g'][0,:]),label = r"$b_r$")
    plt.plot(r[0,:],np.abs(bphi2['g'][0,:]),label = r"$b_\phi$")
    plt.plot(r[0,:],np.abs(bz['g'][0,:]),label = r"$b_z$")
    plt.plot(r[0,:],np.abs(p2['g'][0,:]),label = r"$p$")
    plt.legend()
    plt.savefig(f"/Users/luis_lu/Desktop/PPPL/mri_cylindrical/plots/EigenFunc_{insulating}_1e{np.log10(Re)}_Rm={Rm}_m={m}_n={n}_Nr={Nr}.png",dpi=300)

if __name__ == "__main__":
    main()

