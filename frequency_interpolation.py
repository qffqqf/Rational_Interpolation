import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Model definition
nDOF = 5
nFreq = 1000
k = 6
m = 1
damping = 0.01
nDOF_check = 3
freq_array = np.linspace(0,1,nFreq)
K = 2*k*np.diag(np.ones(nDOF)) - k*np.diag(np.ones(nDOF-1),-1) - k*np.diag(np.ones(nDOF-1),1)
M = m*np.eye(nDOF) 
K = K*(1+damping*1j)
F = np.zeros(nDOF)
F[0] = 1

# Solve the FOM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F)
u_FOM = U_FOM[nDOF_check,:]

# Get reference eigenval
eigval = linalg.eigvals(K, M)
eigfreq = np.sqrt(eigval)/2/np.pi
print(eigfreq)

# Hyper parameters
nSamp = 2
nFP = 1
fp_tol = 1e-2

# Greedy search of samples
Basis = np.zeros([nDOF, nSamp])*1j
Rat_func = np.zeros([nSamp, nFreq])*1j
for iSamp in range(nSamp):
    shift = 0
    Phi = F.copy()
    for iFP in range(nFP):
        # Solve for a new base
        Phi_old = Phi.copy()
        if iSamp > 0:
            Basis_given = Basis[:,0:iSamp]
            D = K - (2*np.pi*shift)**2*M
            D_red = np.matmul(np.matmul(Basis_given.H, D),Basis_given)
            F_red = np.matmul(Basis_given.H, F)
            H_red = np.linalg.solve(D_red, F_red)
            F_int = np.matmul(np.matmul(D, Basis_given),H_red)
            Phi = np.linalg.solve(D, F - F_int)
        else:
            Phi = np.linalg.solve(D, F)
        # Solve for a new shift
        shift_old = shift.copy()
        k_scal = np.matmul(np.matmul(Phi.H, K),Phi)
        m_scal = np.matmul(np.matmul(Phi.H, M),Phi)
        shift = -np.conjugate(np.sqrt(k_scal/m_scal)/2/np.pi)
        # Check relative difference
        diff_fp = np.linalg.norm(Phi-Phi_old)/np.linalg.norm(Phi) + np.abs(shift-shift_old)/np.abs(shift)
        if diff_fp < fp_tol:
            break
    # Add obtained basis and solve for rational function
    Basis[:,iSamp] = Phi
    for iFreq in range(nFreq):
        omega = 2*np.pi*freq_array[iFreq]
        k_scal = np.matmul(np.matmul(Phi.H, K),Phi)
        m_scal = np.matmul(np.matmul(Phi.H, M),Phi)
        f_scal = np.matmul(Phi.H, F)
        Rat_func[iSamp,iFreq] = f_scal/(k_scal - omega**2*m_scal)

# Solve the ROM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F)
u_FOM = U_FOM[nDOF_check,:]




# Plot FRF
#plt.semilogy(freq_array, np.abs(u_FOM))
#plt.semilogy(np.abs(eigfreq), np.mean(np.abs(u_FOM))*np.ones(nDOF), 'o')
#plt.show()