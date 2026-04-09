"""
SM: Shell radii comparison + distance-dependent ATT/REP histogram
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import genlaguerre
from math import factorial

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

beta = 2.0
phi_p = beta
beta_phi = np.sinh(phi_p)/phi_p*beta
omega_phi = 1.0/np.cosh(phi_p/2.0)
sigma2 = beta_phi

# Polar basis
def build_states(N):
    states=[]; E=1
    while len(states)<N:
        for n_r in range(E):
            abs_m=E-1-2*n_r
            if abs_m<0: continue
            if abs_m==0: states.append((n_r,0))
            else: states.append((n_r,abs_m)); states.append((n_r,-abs_m))
            if len(states)>=N: break
        E+=1
    return states[:N]

def polar_wf(n_r,m,x,y):
    r2=x**2+y**2;r=np.sqrt(r2);am=abs(m);th=np.arctan2(y,x)
    norm=np.sqrt(2.0*factorial(n_r)/factorial(n_r+am))/np.sqrt(np.pi)
    if m==0:norm/=np.sqrt(2.0)
    L=genlaguerre(n_r,am);rad=r**am*L(r2)*np.exp(-r2/2.0)
    if m>0:ang=np.cos(m*th)
    elif m<0:ang=np.sin(am*th)
    else:ang=1.0
    return norm*rad*ang

def find_vtotal_min(N):
    def Vt(v):
        pos=v.reshape(N,2);Vh=0.5*omega_phi**2*np.sum(pos**2)
        d2=np.sum((pos[:,None,:]-pos[None,:,:])**2,axis=2)
        K=np.exp(-d2/(2.0*sigma2));s,ld=np.linalg.slogdet(K)
        return Vh+(-ld/beta_phi if s>0 else 1e10)
    def Vg(v):
        eps=1e-6;f0=Vt(v);g=np.empty_like(v)
        for i in range(len(v)):vp=v.copy();vp[i]+=eps;g[i]=(Vt(vp)-f0)/eps
        return g
    bf,bx=np.inf,None
    ns=20 if N<=15 else 8
    for seed in range(ns):
        rng=np.random.RandomState(seed);x0=np.zeros((N,2));idx=0
        ms=int(np.ceil(np.sqrt(2*N)));r=0.0
        for s in range(ms+1):
            ni=s+1
            if idx+ni>N:ni=N-idx
            if ni<=0:break
            if s==0:x0[idx]=[0,0];idx+=1;r=0.7
            else:
                r+=0.55+rng.randn()*0.03
                for k in range(ni):
                    a=2*np.pi*k/ni+rng.randn()*0.05+seed*0.3
                    x0[idx]=[r*np.cos(a),r*np.sin(a)];idx+=1
            if idx>=N:break
        res=minimize(Vt,x0.ravel(),jac=Vg,method='L-BFGS-B',
                     options={'maxiter':30000,'ftol':1e-15})
        if res.fun<bf:bf,bx=res.fun,res.x.reshape(N,2)
    return bx[np.argsort(np.linalg.norm(bx,axis=1))]

def find_slater_max(N):
    states=build_states(N)
    def smat(pos):
        S=np.empty((N,N))
        for i,(nr,m) in enumerate(states):S[i]=polar_wf(nr,m,pos[:,0],pos[:,1])
        return S
    def neg_ld(v):
        s,ld=np.linalg.slogdet(smat(v.reshape(N,2)));return -2*ld if s!=0 else 1e30
    def neg_g(v):
        eps=1e-6;f0=neg_ld(v);g=np.empty_like(v)
        for i in range(len(v)):vp=v.copy();vp[i]+=eps;g[i]=(neg_ld(vp)-f0)/eps
        return g
    bf,bx=np.inf,None
    ns=20 if N<=15 else 8
    for seed in range(ns):
        rng=np.random.RandomState(seed);x0=np.zeros((N,2));idx=0
        ms=int(np.ceil(np.sqrt(2*N)));r=0.0
        for s in range(ms+1):
            ni=s+1
            if idx+ni>N:ni=N-idx
            if ni<=0:break
            if s==0:x0[idx]=[0,0];idx+=1;r=0.7
            else:
                r+=0.55+rng.randn()*0.03
                for k in range(ni):
                    a=2*np.pi*k/ni+rng.randn()*0.05+seed*0.3
                    x0[idx]=[r*np.cos(a),r*np.sin(a)];idx+=1
            if idx>=N:break
        res=minimize(neg_ld,x0.ravel(),jac=neg_g,method='L-BFGS-B',
                     options={'maxiter':30000,'ftol':1e-14})
        if res.fun<bf:bf,bx=res.fun,res.x.reshape(N,2)
    return bx[np.argsort(np.linalg.norm(bx,axis=1))]

def get_shell_radii(pc, N):
    r = np.linalg.norm(pc, axis=1)
    shells = []; start=0
    for i in range(1,N):
        if r[i]-r[i-1]>0.15:
            shells.append(np.mean(r[start:i])); start=i
    shells.append(np.mean(r[start:]))
    return shells

# ═══════════════════════════════════════════════════════════════
#  FIG 5: Shell radii vs N
# ═══════════════════════════════════════════════════════════════
print("=== Shell radii comparison ===")
Ns = [3, 6, 10, 15, 21, 28, 36, 45, 55]
fig5, ax5 = plt.subplots(1, 1, figsize=(5, 3.5))

for N in Ns:
    print(f"  N={N}", flush=True)
    pc_v = find_vtotal_min(N)
    pc_s = find_slater_max(N)
    r_v = get_shell_radii(pc_v, N)
    r_s = get_shell_radii(pc_s, N)
    for i, (rv, rs) in enumerate(zip(r_v, r_s)):
        ax5.plot(N, rv, 'bs', ms=4, zorder=5)
        ax5.plot(N, rs, 'r^', ms=4, zorder=5)

from matplotlib.lines import Line2D
leg5 = [Line2D([0],[0],marker='s',color='w',markerfacecolor='b',ms=5,label=r'$V_{\rm total}$ min'),
        Line2D([0],[0],marker='^',color='w',markerfacecolor='r',ms=5,label=r'$|\Psi_0|$ max')]
ax5.legend(handles=leg5, fontsize=9)
ax5.set_xlabel(r'$N$')
ax5.set_ylabel(r'Shell radius $/\, a_0$')
ax5.set_xticks(Ns)

out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig5.savefig(f'{out}\\fig_SM_shell_radii.pdf', dpi=600, bbox_inches='tight')
fig5.savefig(f'{out}\\fig_SM_shell_radii.png', dpi=300, bbox_inches='tight')
print("Saved fig_SM_shell_radii")

# ═══════════════════════════════════════════════════════════════
#  FIG 6: Distance-dependent ATT/REP histogram for N=55
# ═══════════════════════════════════════════════════════════════
print("=== Distance histogram N=55 ===")
N = 55
pc = find_vtotal_min(N)
d2_pc = np.sum((pc[:,None,:]-pc[None,:,:])**2, axis=2)
K = np.exp(-d2_pc/(2.0*sigma2))
Kinv = np.linalg.inv(K)

dists_att, dists_rep = [], []
for a in range(N):
    for b in range(a+1,N):
        coeff=Kinv[a,b]*K[a,b]
        f=(2.0/sigma2)*(pc[b]-pc[a])*coeff/beta_phi
        dr=pc[b]-pc[a]; dist=np.linalg.norm(dr)
        dot=np.dot(f,dr/dist)
        if dot>0: dists_att.append(dist)
        else: dists_rep.append(dist)

dists_att=np.array(dists_att); dists_rep=np.array(dists_rep)
max_d=max(dists_att.max(),dists_rep.max())
bins=np.linspace(0, max_d*1.01, 15)
centers=0.5*(bins[:-1]+bins[1:]); width=bins[1]-bins[0]

na_hist=[((dists_att>=bins[i])&(dists_att<bins[i+1])).sum() for i in range(len(bins)-1)]
nr_hist=[((dists_rep>=bins[i])&(dists_rep<bins[i+1])).sum() for i in range(len(bins)-1)]

fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
plt.subplots_adjust(hspace=0.08)

ax6a.bar(centers-width*0.2, na_hist, width*0.4, color='#CC0000', alpha=0.7, label='Attractive')
ax6a.bar(centers+width*0.2, nr_hist, width*0.4, color='#2255CC', alpha=0.7, label='Repulsive')
ax6a.set_ylabel('Number of pairs')
ax6a.legend(fontsize=9)
ax6a.set_title(rf'$N=55$, $\varphi=2$', fontsize=11)

# Fraction attractive
frac_att = [na_hist[i]/(na_hist[i]+nr_hist[i]) if (na_hist[i]+nr_hist[i])>0 else 0
            for i in range(len(bins)-1)]
ax6b.bar(centers, frac_att, width*0.8, color='#CC0000', alpha=0.5)
ax6b.axhline(0.5, color='grey', ls='--', lw=0.8)
ax6b.set_xlabel(r'Pair distance $/\, a_0$')
ax6b.set_ylabel(r'Attractive fraction')
ax6b.set_ylim(0, 1)

fig6.savefig(f'{out}\\fig_SM_distance_histogram.pdf', dpi=600, bbox_inches='tight')
fig6.savefig(f'{out}\\fig_SM_distance_histogram.png', dpi=300, bbox_inches='tight')
print("Saved fig_SM_distance_histogram")
print("Done")
