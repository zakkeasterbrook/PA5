"""Robust randomized monotone stopping from mean score checkpoints.
LP discovers candidates; exact rational Bellman bounds certify them.
No model calls. All economic conclusions conditional on the stated score model.
"""
from fractions import Fraction as Q
from collections import defaultdict
from pathlib import Path
import argparse,csv,json,itertools,time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parent

def exact_bound(F,a,costs,means,K,kind):
    """Dual bound in exact rational arithmetic on all monotone paths."""
    x=[Q(i,K) for i in range(K+1)]
    v=[-a[-1]*z for z in x]
    for t in range(len(costs)-2,-1,-1):
        if kind=='save':
            v=[(costs[t+1]-costs[t])*F[t][j]-a[t]*z+min(v[j:]) for j,z in enumerate(x)]
        elif kind=='loss':
            v=[max((x[k]-z)*F[t][j]+v[k] for k in range(j,K+1))-a[t]*z for j,z in enumerate(x)]
        else: raise ValueError(kind)
    return (min(v) if kind=='save' else max(v))+sum(ai*m for ai,m in zip(a,means))

def solve(costs,means,K,eps):
    T=len(costs); M=K+1; rawB=costs[-1]; costs=[c/rawB for c in costs]
    assert T>=2 and K>=1 and len(means)==T and 0<=eps<=1
    assert costs[0]>=0 and all(a<b for a,b in zip(costs,costs[1:]))
    assert 0<=means[0]<=means[-1]<=1 and all(a<=b for a,b in zip(means,means[1:]))
    nf=(T-1)*M; ao=nf; bo=ao+T; so=bo+T; lo=so+T*M; A=lo+T*M; B=A+1; n=B+1
    f=lambda t,j:t*M+j
    s=lambda t,j:so+t*M+j
    l=lambda t,j:lo+t*M+j
    row=[];col=[];dat=[];rhs=[]
    def le(entries,bound=0.):
        i=len(rhs);rhs.append(bound)
        for j,v in entries:
            if v: row.append(i);col.append(j);dat.append(float(v))
    for t in range(T-1):
        for j in range(M):
            if j<K: le([(f(t,j),1),(f(t,j+1),-1)])
            if t<T-2: le([(f(t,j),1),(f(t+1,j),-1)])
    for t in range(T-1):
        d=costs[t+1]-costs[t]
        for j in range(M):
            z=Q(j,K)
            for k in range(j,M):
                le([(s(t,j),1),(s(t+1,k),-1),(f(t,j),-d),(ao+t,z)])
                le([(l(t,j),-1),(l(t+1,k),1),(f(t,j),Q(k-j,K)),(bo+t,-z)])
    for j in range(M):
        z=Q(j,K)
        le([(s(T-1,j),1),(ao+T-1,z)])
        le([(l(T-1,j),-1),(bo+T-1,-z)])
        le([(A,1),(s(0,j),-1)])
        le([(B,-1),(l(0,j),1)])
    le([(B,1)]+[(bo+t,m) for t,m in enumerate(means)],float(eps))
    obj=np.zeros(n); obj[A]=-1
    for t,m in enumerate(means):obj[ao+t]=-float(m)
    mat=coo_matrix((dat,(row,col)),shape=(len(rhs),n)).tocsr()
    start=time.perf_counter()
    res=linprog(obj,A_ub=mat,b_ub=np.array(rhs),bounds=[(0,1)]*nf+[(None,None)]*(n-nf),method='highs')
    seconds=time.perf_counter()-start
    if not res.success:raise RuntimeError(res.message)
    D=10**9
    q=lambda x:Q(round(float(x)*D),D)
    F=[]
    for t in range(T-1):
        r=[]
        for j in range(M):
            r.append(max(Q(0),min(Q(1),q(res.x[f(t,j)])),r[-1] if r else Q(0),F[-1][j] if F else Q(0)))
        F.append(r)
    a=[q(z) for z in res.x[ao:ao+T]]; b=[q(z) for z in res.x[bo:bo+T]]
    loss=exact_bound(F,b,costs,means,K,'loss')
    scale=Q(1) if loss<=eps else eps/loss
    F=[[scale*z for z in r] for r in F];a=[scale*z for z in a];b=[scale*z for z in b]
    save=exact_bound(F,a,costs,means,K,'save');loss=exact_bound(F,b,costs,means,K,'loss')
    assert loss<=eps
    assert all(0<=z<=1 for r in F for z in r)
    assert all(u<=v for r in F for u,v in zip(r,r[1:]))
    assert all(u<=v for r,sr in zip(F,F[1:]) for u,v in zip(r,sr))
    return dict(estimated_lp_savings=float(-res.fun),certified_savings=save,certified_loss=loss,cap=rawB*(1-save),F=F,a=a,b=b,seconds=seconds,variables=n,constraints=len(rhs))

def fixed(costs,means,eps):
    """Exact optimal score-independent randomized cap."""
    target=means[-1]-eps
    best=min((c,[(i,Q(1))]) for i,c in enumerate(costs) if means[i]>=target)
    for i,j in itertools.combinations(range(len(costs)),2):
        if means[i]<target<means[j]:
            p=(target-means[i])/(means[j]-means[i]); cand=((1-p)*costs[i]+p*costs[j],[(i,1-p),(j,p)])
            if cand[0]<best[0]:best=cand
    return best

def uniform(costs,means,K):
    """Mean-identifiable uniform random target savings; sharp worst loss."""
    inc=[means[0]]+[b-a for a,b in zip(means,means[1:])]
    if means[-1]==0:return Q(0),Q(0)
    v=[K*z/means[-1] for z in inc]
    sq=lambda z:(z.numerator//z.denominator)**2+(2*(z.numerator//z.denominator)+1)*(z-(z.numerator//z.denominator))
    loss=means[-1]*(K*K-sum(sq(z) for z in v))/(2*K*K)
    save=sum((b-a)*m for a,b,m in zip(costs,costs[1:],means))
    return save,loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args()
    groups=defaultdict(list)
    with (ROOT/'metr_points.csv').open() as f:
        for r in csv.DictReader(f):
            if r['family']=='Earthborne Rangers':groups[r['model']].append((Q(r['cost_usd']),Q(r['mean_best_score'])))
    results=[]
    for model,rows in groups.items():
        costs,means=map(list,zip(*rows));K=21
        us,ul=uniform(costs,means,K)
        print(model,'uniform',float(us),float(ul),'n',len(rows),flush=True)
        for eps in ([Q('0.01')] if args.quick else [Q('0.005'),Q('0.01'),Q('0.02'),Q('0.05'),Q('0.10')]):
            z=solve(costs,means,K,eps); fc,w=fixed(costs,means,eps)
            r=dict(model=model,epsilon=eps,end_cap=costs[-1],fixed_cap=fc,fixed_policy=w,uniform_savings=us,uniform_worst_loss=ul,**z)
            results.append(r)
            print('eps',float(eps),'fixed',float(fc),'robust',float(z['cap']),'loss',float(z['certified_loss']),'lp',z['estimated_lp_savings'],'cert',float(z['certified_savings']),'seconds',z['seconds'],flush=True)
            (ROOT/'results.json').write_text(json.dumps(results,default=str,indent=2)+'\n')
if __name__=='__main__':main()
