"""Exact-rational two-variable optimizer for the two-checkpoint robust policy.
No SciPy or floating-point calculation is needed to construct the policy.
The independent numerical LP is used only as a cross-check.
"""
from fractions import Fraction as Q
from pathlib import Path
from collections import defaultdict
import random,json
from barrier import grid,moment_upper,worst_save,worst_loss,lp_opt

ROOT=Path(__file__).resolve().parent

def clip(poly,A,B,C):
    """Intersect rational convex polygon with A*a+B*b <= C."""
    if not poly:return []
    if A==0 and B==0:return poly if C>=0 else []
    out=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        dp=A*p[0]+B*p[1]-C;dq=A*q[0]+B*q[1]-C
        if dp<=0:out.append(p)
        if (dp<0<dq) or (dq<0<dp):
            t=dp/(dp-dq);out.append(tuple(x+t*(y-x) for x,y in zip(p,q)))
    # Remove adjacent duplicate vertices, including wraparound.
    cleaned=[]
    for p in out:
        if not cleaned or p!=cleaned[-1]:cleaned.append(p)
    if len(cleaned)>1 and cleaned[0]==cleaned[-1]:cleaned.pop()
    return cleaned

def optimize(mu,nu,K,eps):
    """Return globally optimal worst-case saved fraction, f, and minorant.
    Conditions: 0<=mu<nu<=1, 0<eps<nu-mu, uniform K-point subdivisions.
    Exact Fraction arithmetic; uses the no-gain criterion for easy cases.
    """
    mu,nu,eps=map(Q,(mu,nu,eps));d=nu-mu
    assert K>=1 and 0<=mu<nu<=1 and 0<eps<d
    alpha=eps/d;x=grid(K)
    if moment_upper(mu,nu,K)>=mu*d:return alpha,[alpha]*(K+1),(alpha,Q(0))
    # A positive-gain optimum has a nonnegative supporting-line slope.
    Bmax=(1-alpha)/(1-mu);amin=alpha-Bmax*mu
    poly=[(amin,Q(0)),(Q(1),Q(0)),(Q(1),Bmax),(amin,Bmax)]
    poly=clip(poly,Q(1),Q(1),Q(1))
    poly=clip(poly,Q(-1),-mu,-alpha)
    z=mu/nu
    # The worst loss is nu times the upper concave envelope of
    # (1-x) max(0,a+b*x) at z. Each two-point support supplies at most
    # three nontrivial halfplanes, obtained by expanding the two maxima.
    for u in x:
        if u==z:
            A=nu*(1-u);poly=clip(poly,A,A*u,eps)
        if u>=z:continue
        for v in x:
            if v<=z:continue
            A=nu*(1-u)*(v-z)/(v-u);C=nu*(1-v)*(z-u)/(v-u)
            for P,R in [(A,Q(0)),(Q(0),C),(A,C)]:
                poly=clip(poly,P+R,P*u+R*v,eps)
    assert poly
    a,b=max(poly,key=lambda t:t[0]+mu*t[1]);s=a+b*mu
    f=[max(Q(0),a+b*u) for u in x]
    assert all(0<=v<=1 for v in f)
    assert worst_loss(f,mu,nu,K)<=eps
    assert worst_save(f,mu,K)==s
    return s,f,(a,b)

def main():
    rng=random.Random(42611);counts=defaultdict(int);maxerror=0
    for K in (1,2,3,4,7,12,21):
        for _ in range(75):
            n=rng.randrange(1,65);m=rng.randrange(n);mu=Q(m,64);nu=Q(n,64)
            # Force saturation in some cases to exercise the nontrivial polygon solver.
            if _%3==0:nu=Q(1)
            eps=(nu-mu)*Q(rng.randrange(1,100),100)
            s,f,(a,b)=optimize(mu,nu,K,eps)
            assert s>=eps/(nu-mu)
            optimum=lp_opt(mu,nu,K,eps);err=abs(float(s)-optimum);maxerror=max(maxerror,err)
            assert err<1e-8,(mu,nu,K,eps,s,optimum,a,b)
            counts['independent_exact_optimizer_comparisons']+=1
            counts['strict_improvement_cases' if s>eps/(nu-mu) else 'no_improvement_cases']+=1
    examples=[]
    for K,mu,nu,eps in [(21,Q(41,42),Q(1),Q(1,84)),(1,Q(3,5),Q(9,10),Q(3,100))]:
        s,f,(a,b)=optimize(mu,nu,K,eps)
        examples.append(dict(K=K,mu=str(mu),nu=str(nu),eps=str(eps),fixed_fraction=str(eps/(nu-mu)),optimal_adaptive_fraction=str(s),worst_loss=str(worst_loss(f,mu,nu,K)),a=str(a),b=str(b),policy=list(map(str,f))))
    result=dict(counts=counts,max_independent_LP_error=maxerror,examples=examples)
    print(json.dumps(result,indent=2));(ROOT/'exact_policy_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
