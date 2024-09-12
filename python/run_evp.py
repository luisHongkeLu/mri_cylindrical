from mri_lib import mri_evp_24
import scipy,numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt

Rm = 4
B0 = 0.1
Re = 1e3
r2 = 3
O2 = 0
insulating = True
baseflow_name = "base flow, rm=4 B_0=0.2, z=-1 to 1"
m = 1
n=2
h=4
m_max = 4
Nr = 256
num=0
Nphi = 2 * m_max + 2
kz = n*np.pi/h
data = np.genfromtxt(f"/Users/luis_lu/Desktop/PPPL/mri_cylindrical/python/{baseflow_name}.txt")
r_data = data[0:,0]
Omega_data = data[0:,1]
f_pchip = scipy.interpolate.PchipInterpolator(r_data,Omega_data)
growth = []
count = 0
#solve the evp
solver, u, v, p, br,bphi, dist,coords,annulus = mri_evp_24(Re,Rm,r2,B0,f_pchip,m,kz,Nr,Nphi,insulating)
evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
s = evals[np.argsort(-evals.imag)][0]
#if some threshold that the abs value of eigenvalue should be smaller than some number
#some times dedalus will get some numerically spurious mode 
while np.abs(s)>10:
    count+=1
    print("recalculating")
    #solve the evp again
    solver, u, v, p, br,bphi, dist,coords,annulus = mri_evp_24(Re,Rm,r2,B0,f_pchip,m,kz,Nr+count*100,Nphi,insulating)
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    s = evals[np.argsort(-evals.imag)][0]
max_eigenvalue =evals[np.argsort(-evals.imag)][0]
#find the correct eigenmode index
solver.set_state(np.argmin(np.abs(solver.eigenvalues - max_eigenvalue)), 0)
#get all the eigenfunctions
bz = dist.Field(name='bz', bases=annulus)
w = dist.Field(name='w', bases=annulus)
phi,r = dist.local_grids(annulus[0], annulus[1])
r_1 = dist.Field(name='r_1', bases=annulus[1])
dr = lambda A: d3.Differentiate(A, coords['r'])
r_1['g'] = r
bz = -(1/r_1*(dr(r_1*br))+1/r_1*1j*m*bphi)/(1j*kz)
w = -(1/r_1*(dr(r_1*u))+1/r_1*1j*m*v)/(1j*kz)
#save the data
f = open(f'./s_output.csv', 'a')
f.write(f"{Rm},{B0},{s.imag},{s.real}\n")
f.close()
g = open(f'./br_w.csv', 'a')
g.write(f"{Rm},{B0},{np.max(np.abs(br['g'][0,:]))},{np.max(np.abs(w['g'][0,:]))},{np.max(np.abs(br['g'][0,:]))/np.max(np.abs(w['g'][0,:]))}\n")
g.close()



