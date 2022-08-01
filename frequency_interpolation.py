import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Model definition
nDOF = 40
nFreq = 1000
k = 1000
m = 1
damping = 0.03
nDOF_check = 3
freq_array = np.linspace(0,5,nFreq)
K = 2*k*np.diag(np.ones(nDOF)) - k*np.diag(np.ones(nDOF-1),-1) - k*np.diag(np.ones(nDOF-1),1)
M = m*np.eye(nDOF) 
K = K*(1+damping*1j)
F = np.zeros([nDOF,1])
F[0] = 1
F_sim = np.ones([nDOF,1])

# Solve the FOM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F[:,0])
u_FOM = U_FOM[nDOF_check,:]

# Get reference eigenval
[eigval, eigvec] = linalg.eig(K, M, right=True)
eigfreq = np.sqrt(eigval)/2/np.pi
#print(np.diag(np.matmul(np.matmul(eigvec.conj().T, M),eigvec)))
#print(np.sqrt(np.diag(np.matmul(np.matmul(eigvec.conj().T, K),eigvec)))/2/np.pi)

# Hyper parameters
nSamp = 3
nFP = 100
fp_tol = 1e-2

# Greedy search of samples
Basis = np.zeros([nDOF, nSamp])*1j
Rat_func = np.zeros([nSamp, nFreq])*1j
for iSamp in range(nSamp):
    shift = 0
    Phi = np.zeros([nDOF,1])*1j
    for iFP in range(nFP):
        print(iFP)
        # Solve for a new base
        Phi_old = Phi.copy()
        D = K - (2*np.pi*shift)**2*M
        if iFP < 1:
            D = np.matmul(D.conj().T, D)
        if iSamp > 0:
            if iFP < 1:
                Basis_given = Basis[:,0:iSamp]
                D_red = np.matmul(np.matmul(Basis_given.conj().T, D),Basis_given)
                F_red = np.matmul(Basis_given.conj().T, F_sim)
                H_red = np.linalg.solve(D_red, F_red)
                F_int = np.matmul(D, np.matmul(Basis_given, H_red))
                Phi = np.linalg.solve(D, F_sim - F_int)
            else:
                Basis_given = np.concatenate((Basis[:,0:iSamp], Phi), axis=1) 
                D_red = np.matmul(np.matmul(Basis_given.conj().T, D),Basis_given)
                F_red = np.matmul(Basis_given.conj().T, F)
                H_red = np.linalg.solve(D_red, F_red)
                F_int = np.matmul(D, np.matmul(Basis[:,0:iSamp], H_red[0:iSamp]))
                Phi = np.linalg.solve(D, F - F_int)
        else:
            if iFP < 1:
                Phi = np.linalg.solve(D, F_sim)
            else:
                Phi = np.linalg.solve(D, F)
        Phi = Phi/np.linalg.norm(Phi)
        # shift verification
        weights = np.matmul(eigvec.conj().T, Phi)
        weights = weights.conj()*weights
        shift_pre = np.sum(eigval*weights)/np.sum(weights)
        #print(["weights: ", weights, "shift_pre", shift_pre])
        #print(["eigval: ", np.abs(eigval)])
        # Solve for a new shift
        shift_old = shift
        k_scal = np.matmul(np.matmul(Phi.conj().T, K),Phi)
        m_scal = np.matmul(np.matmul(Phi.conj().T, M),Phi)
        shift = -np.conjugate(np.sqrt(k_scal/m_scal)/2/np.pi)
        #print("old : %.5e, new : %.5e" % (abs(shift_old), abs(shift)))
        # Check relative difference
        diff_fp = np.linalg.norm(Phi-Phi_old)/np.linalg.norm(Phi) + np.abs(shift-shift_old)/np.abs(shift)
        if diff_fp < fp_tol:
            print("#########################################################")
            break
    # Add obtained basis and solve for rational function
    Basis[:,iSamp] = Phi[:,0]
    for iFreq in range(nFreq):
        omega = 2*np.pi*freq_array[iFreq]
        k_scal = np.matmul(np.matmul(Phi.conj().T, K),Phi)
        m_scal = np.matmul(np.matmul(Phi.conj().T, M),Phi)
        f_scal = np.matmul(Phi.conj().T, F)
        Rat_func[iSamp,iFreq] = f_scal/(k_scal - omega**2*m_scal)

# Solve the ROM
print(np.matmul(Basis.conj().T, Basis))
U_ROM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    D = K - omega**2*M
    D_red = np.matmul(np.matmul(Basis.conj().T, D),Basis)
    F_red = np.matmul(Basis.conj().T, F)
    H_red = np.linalg.solve(D_red, F_red)
    U_ROM[:,iFreq] = np.matmul(Basis, H_red[:,0])
u_ROM = U_ROM[nDOF_check,:]

# Plot FRF
fig = plt.figure(1)
plt.semilogy(freq_array, np.abs(u_FOM))
plt.semilogy(freq_array, np.abs(u_ROM))

fig = plt.figure(2)
plt.semilogy(freq_array, np.abs(u_FOM-u_ROM)/np.abs(u_FOM))
plt.show()