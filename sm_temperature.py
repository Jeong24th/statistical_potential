"""
SM: Temperature evolution figures
1) V_total contour for N=6 at 4 temperatures
2) Strongest bond for N=55 at 4 temperatures
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

betas_N6 = [0.5, 1.0, 2.0, 3.0]
betas_N55 = [0.5, 1.0, 2.0, 3.0]

def get_params(beta):
    phi=beta; bp=np.sinh(phi)/phi*beta; op=1.0/np.cosh(phi/2.0); return bp,op,bp

def find_min(N, beta):
    bp,op,s2 = get_params(beta)
    def Vt(v):
        pos=v.reshape(N,2); Vh=0.5*op**2*np.sum(pos**2)
        d2=np.sum((pos[:,None,:]-pos[None,:,:])**2,axis=2)
        K=np.exp(-d2/(2.0*s2)); s,ld=np.linalg.slogdet(K)
        return Vh+(-ld/bp if s>0 else 1e10)
    def Vg(v):
        eps=1e-6;f0=Vt(v);g=np.empty_like(v)
        for i in range(len(v)):vp=v.copy();vp[i]+=eps;g[i]=(Vt(vp)-f0)/eps
        return g
    bf,bx=np.inf,None
    ns=100
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
    return bx[np.argsort(np.linalg.norm(bx,axis=1))], bp, op, s2

def compute_forces(pc, N, s2, bp):
    d2=np.sum((pc[:,None,:]-pc[None,:,:])**2,axis=2)
    K=np.exp(-d2/(2.0*s2)); Kinv=np.linalg.inv(K)
    forces={}; fm=np.zeros((N,N)); fa=np.zeros((N,N),dtype=bool)
    for a in range(N):
        for b in range(a+1,N):
            coeff=Kinv[a,b]*K[a,b]
            f=(2.0/s2)*(pc[b]-pc[a])*coeff/bp
            mag=np.linalg.norm(f); dr=pc[b]-pc[a]
            dot=np.dot(f,dr/np.linalg.norm(dr)); att=dot>0
            forces[(a,b)]={'mag':mag,'attractive':att}
            fm[a,b]=mag;fm[b,a]=mag;fa[a,b]=att;fa[b,a]=att
    return forces, fm, fa

# ═══════════════════════════════════════════════════════════════
# FIG 3: V_total contour + forces for N=6, 4 temperatures
# ═══════════════════════════════════════════════════════════════
print("=== N=6 temperature evolution ===")
fig3, axes3 = plt.subplots(1, 4, figsize=(9, 2.5))
plt.subplots_adjust(wspace=0.08, left=0.04, right=0.98, bottom=0.15, top=0.88)

for j, beta in enumerate(betas_N6):
    N=6; ax=axes3[j]
    print(f"  varphi={beta}", flush=True)
    pc,bp,op,s2 = find_min(N, beta)
    forces,_,_ = compute_forces(pc, N, s2, bp)

    # Conditional map
    vary=0; gn=100; gr=3.2
    xg=np.linspace(-gr,gr,gn); yg=np.linspace(-gr,gr,gn)
    Xg,Yg=np.meshgrid(xg,yg)
    Vg=np.empty((gn,gn))
    for iy in range(gn):
        for ix in range(gn):
            p=pc.copy();p[vary]=[xg[ix],yg[iy]]
            Vh=0.5*op**2*np.sum(p**2)
            d2=np.sum((p[:,None,:]-p[None,:,:])**2,axis=2)
            K=np.exp(-d2/(2.0*s2));s,ld=np.linalg.slogdet(K)
            Vg[iy,ix]=Vh+(-ld/bp if s>0 else 1e10)
    Vg-=np.min(Vg)

    vmax=np.percentile(Vg,88); lvl=np.linspace(0,vmax,18)
    ax.contourf(Xg,Yg,Vg,levels=lvl,cmap='RdYlBu_r',extend='max',alpha=0.45)
    ax.contour(Xg,Yg,Vg,levels=lvl[::3],colors='k',linewidths=0.15,alpha=0.2)

    fmax=max(v['mag'] for v in forces.values())
    for (a,b),v in forces.items():
        col='#CC0000' if v['attractive'] else '#2255CC'
        rel=np.sqrt(v['mag']/fmax)
        lw=0.2+1.6*rel
        ax.plot([pc[a,0],pc[b,0]],[pc[a,1],pc[b,1]],color=col,lw=lw,alpha=0.3+0.6*rel,zorder=3)
    for a in range(N):
        ax.plot(pc[a,0],pc[a,1],'*',color='black',ms=5,markeredgecolor='black',markeredgewidth=0.2,zorder=6)

    ax.set_xlim(-gr,gr);ax.set_ylim(-gr,gr);ax.set_aspect('equal')
    ax.set_title(rf'$\varphi={beta}$',fontsize=10)
    if j==0: ax.set_ylabel(r'$y/a_0$')
    else: ax.set_yticklabels([])
    ax.set_xlabel(r'$x/a_0$')

fig3.text(0.02,0.95,r'$N=6$',fontsize=11,fontweight='bold',va='top')
out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig3.savefig(f'{out}\\fig_SM_temp_N6.pdf',dpi=600,bbox_inches='tight')
fig3.savefig(f'{out}\\fig_SM_temp_N6.png',dpi=300,bbox_inches='tight')
print("Saved fig_SM_temp_N6")

# ═══════════════════════════════════════════════════════════════
# FIG 4: Strongest bond for N=55, 4 temperatures
# ═══════════════════════════════════════════════════════════════
print("=== N=55 temperature evolution ===")
fig4, axes4 = plt.subplots(1, 4, figsize=(9, 2.5))
plt.subplots_adjust(wspace=0.08, left=0.04, right=0.98, bottom=0.15, top=0.88)

for j, beta in enumerate(betas_N55):
    N=55; ax=axes4[j]
    print(f"  varphi={beta}", flush=True)
    pc,bp,op,s2 = find_min(N, beta)
    forces,fm,fa = compute_forces(pc, N, s2, bp)

    # Strongest bonds
    bonds=set()
    for a in range(N):
        mags=fm[a].copy();mags[a]=0;b=np.argmax(mags)
        bonds.add((min(a,b),max(a,b)))

    lim=max(np.max(np.abs(pc))*1.2,2.0)
    fmax_b=max(forces[(a2,b2)]['mag'] for a2,b2 in bonds)
    for (a,b) in bonds:
        v=forces[(a,b)]
        col='#CC0000' if v['attractive'] else '#2255CC'
        rel=v['mag']/fmax_b; lw=0.5+2.0*rel; alpha=0.5+0.4*rel
        ax.plot([pc[a,0],pc[b,0]],[pc[a,1],pc[b,1]],color=col,lw=lw,alpha=alpha,zorder=3)
    for a in range(N):
        ax.plot(pc[a,0],pc[a,1],'*',color='black',ms=3,markeredgecolor='black',markeredgewidth=0.15,zorder=6)

    n_att=sum(1 for a2,b2 in bonds if forces[(a2,b2)]['attractive'])
    ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect('equal')
    ax.set_title(rf'$\varphi={beta}$',fontsize=10)
    ax.text(0.95,0.05,rf'{n_att}a/{len(bonds)-n_att}r',transform=ax.transAxes,
            fontsize=7,va='bottom',ha='right',color='#666666')
    if j==0: ax.set_ylabel(r'$y/a_0$')
    else: ax.set_yticklabels([])
    ax.set_xlabel(r'$x/a_0$')

fig4.text(0.02,0.95,r'$N=55$',fontsize=11,fontweight='bold',va='top')
fig4.savefig(f'{out}\\fig_SM_temp_N55_strongest.pdf',dpi=600,bbox_inches='tight')
fig4.savefig(f'{out}\\fig_SM_temp_N55_strongest.png',dpi=300,bbox_inches='tight')
print("Saved fig_SM_temp_N55_strongest")
print("Done")
