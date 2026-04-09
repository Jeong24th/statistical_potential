"""
SM: V_total + force lines AND strongest bond for multiple N (closed shells)
3x3 grid each, at varphi=2.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

beta = 2.0
phi_p = beta
beta_phi = np.sinh(phi_p)/phi_p*beta
omega_phi = 1.0/np.cosh(phi_p/2.0)
sigma2 = beta_phi

Ns = [3, 6, 10, 15, 21, 28, 36, 45, 55]

def find_min_and_forces(N):
    def V_total(v):
        pos = v.reshape(N,2)
        Vh = 0.5*omega_phi**2*np.sum(pos**2)
        d2 = np.sum((pos[:,None,:]-pos[None,:,:])**2, axis=2)
        K = np.exp(-d2/(2.0*sigma2))
        s,ld = np.linalg.slogdet(K)
        return Vh + (-ld/beta_phi if s>0 else 1e10)
    def V_grad(v):
        eps=1e-6; f0=V_total(v); g=np.empty_like(v)
        for i in range(len(v)): vp=v.copy(); vp[i]+=eps; g[i]=(V_total(vp)-f0)/eps
        return g

    best_f, best_x = np.inf, None
    ns = 20 if N<=15 else 8
    for seed in range(ns):
        rng=np.random.RandomState(seed); x0=np.zeros((N,2)); idx=0
        ms=int(np.ceil(np.sqrt(2*N))); r=0.0
        for s in range(ms+1):
            ni=s+1
            if idx+ni>N: ni=N-idx
            if ni<=0: break
            if s==0: x0[idx]=[0,0]; idx+=1; r=0.7
            else:
                r+=0.55+rng.randn()*0.03
                for k in range(ni):
                    a=2*np.pi*k/ni+rng.randn()*0.05+seed*0.3
                    x0[idx]=[r*np.cos(a),r*np.sin(a)]; idx+=1
            if idx>=N: break
        res=minimize(V_total,x0.ravel(),jac=V_grad,method='L-BFGS-B',
                     options={'maxiter':30000,'ftol':1e-15})
        if res.fun<best_f: best_f,best_x=res.fun,res.x.reshape(N,2)

    pc = best_x[np.argsort(np.linalg.norm(best_x,axis=1))]

    # Forces
    d2_pc = np.sum((pc[:,None,:]-pc[None,:,:])**2, axis=2)
    K = np.exp(-d2_pc/(2.0*sigma2))
    Kinv = np.linalg.inv(K)
    forces = {}
    force_mag_mat = np.zeros((N,N))
    force_att_mat = np.zeros((N,N), dtype=bool)
    for a in range(N):
        for b in range(a+1,N):
            coeff = Kinv[a,b]*K[a,b]
            f = (2.0/sigma2)*(pc[b]-pc[a])*coeff/beta_phi
            mag = np.linalg.norm(f)
            dr = pc[b]-pc[a]
            dot = np.dot(f,dr/np.linalg.norm(dr))
            att = dot>0
            forces[(a,b)] = {'mag':mag, 'attractive':att}
            force_mag_mat[a,b]=mag; force_mag_mat[b,a]=mag
            force_att_mat[a,b]=att; force_att_mat[b,a]=att

    # Strongest bond per particle
    strongest = {}
    for a in range(N):
        mags = force_mag_mat[a].copy(); mags[a]=0
        b = np.argmax(mags)
        strongest[a] = (b, mags[b], force_att_mat[a,b])

    return pc, forces, strongest

# Compute all
all_data = {}
for N in Ns:
    print(f"N={N} ...", flush=True)
    pc, forces, strongest = find_min_and_forces(N)
    all_data[N] = {'pc':pc, 'forces':forces, 'strongest':strongest}
    n_att = sum(1 for v in forces.values() if v['attractive'])
    n_rep = sum(1 for v in forces.values() if not v['attractive'])
    print(f"  {n_att} att, {n_rep} rep")

# ═══════════════════════════════════════════════════════════════
#  PLOT 1: Force lines (3x3)
# ═══════════════════════════════════════════════════════════════
print("Plotting force lines ...", flush=True)
fig1, axes1 = plt.subplots(3, 3, figsize=(7, 7))
plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.02, right=0.98, bottom=0.02, top=0.96)

for idx, N in enumerate(Ns):
    ax = axes1[idx//3, idx%3]
    d = all_data[N]; pc = d['pc']; forces = d['forces']
    lim = max(np.max(np.abs(pc))*1.3, 1.5)

    fmax = max(v['mag'] for v in forces.values())
    threshold = 0.02*fmax if N>10 else 0
    for (a,b),v in forces.items():
        if v['mag']<threshold: continue
        col = '#CC0000' if v['attractive'] else '#2255CC'
        rel = v['mag']/fmax
        lw = 0.3+1.2*rel if N>10 else 0.5+1.5*rel
        alpha = 0.4+0.5*rel
        ax.plot([pc[a,0],pc[b,0]],[pc[a,1],pc[b,1]],
                color=col,lw=lw,alpha=alpha,zorder=3,solid_capstyle='round')
    for a in range(N):
        ax.plot(pc[a,0],pc[a,1],'o',color='black',ms=2.5 if N>15 else 3.5,
                markeredgecolor='black',markeredgewidth=0.3,zorder=6)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.05,0.95,rf'$N={N}$',transform=ax.transAxes,fontsize=9,va='top',fontweight='bold')

out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig1.savefig(f'{out}\\fig_SM_multiN_forces.pdf', dpi=600, bbox_inches='tight')
fig1.savefig(f'{out}\\fig_SM_multiN_forces.png', dpi=300, bbox_inches='tight')
print("Saved fig_SM_multiN_forces")

# ═══════════════════════════════════════════════════════════════
#  PLOT 2: Strongest bond (3x3)
# ═══════════════════════════════════════════════════════════════
print("Plotting strongest bonds ...", flush=True)
fig2, axes2 = plt.subplots(3, 3, figsize=(7, 7))
plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.02, right=0.98, bottom=0.02, top=0.96)

for idx, N in enumerate(Ns):
    ax = axes2[idx//3, idx%3]
    d = all_data[N]; pc = d['pc']; strongest = d['strongest']
    lim = max(np.max(np.abs(pc))*1.3, 1.5)

    bonds = set()
    for a in range(N):
        b,mag,att = strongest[a]
        bonds.add((min(a,b),max(a,b)))

    fmax_b = max(d['forces'][(min(a,strongest[a][0]),max(a,strongest[a][0]))]['mag'] for a in range(N))
    for (a,b) in bonds:
        v = d['forces'][(a,b)]
        col = '#CC0000' if v['attractive'] else '#2255CC'
        rel = v['mag']/fmax_b
        lw = 0.8+2.0*rel
        alpha = 0.5+0.4*rel
        ax.plot([pc[a,0],pc[b,0]],[pc[a,1],pc[b,1]],
                color=col,lw=lw,alpha=alpha,zorder=3,solid_capstyle='round')
    for a in range(N):
        ax.plot(pc[a,0],pc[a,1],'o',color='black',ms=2.5 if N>15 else 3.5,
                markeredgecolor='black',markeredgewidth=0.3,zorder=6)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    n_att_b = sum(1 for a2,b2 in bonds if d['forces'][(a2,b2)]['attractive'])
    n_rep_b = len(bonds)-n_att_b
    ax.text(0.05,0.95,rf'$N={N}$',transform=ax.transAxes,fontsize=9,va='top',fontweight='bold')
    if n_att_b > 0:
        ax.text(0.95,0.05,rf'{n_att_b}a/{n_rep_b}r',transform=ax.transAxes,
                fontsize=7,va='bottom',ha='right',color='#666666')

fig2.savefig(f'{out}\\fig_SM_multiN_strongest.pdf', dpi=600, bbox_inches='tight')
fig2.savefig(f'{out}\\fig_SM_multiN_strongest.png', dpi=300, bbox_inches='tight')
print("Saved fig_SM_multiN_strongest")
print("Done")
