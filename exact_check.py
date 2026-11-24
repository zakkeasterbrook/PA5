"""Exact rational property tests for block bounds and the decision rule.
Positive rational weights are arbitrary within their certified intervals;
this isolates the theorem from floating-point exp and dot products.
"""
from fractions import Fraction as F
from math import isqrt
import random,json


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm2(a): return dot(a,a)
def upper_sqrt(x):
    # An outward rational upper bound on sqrt(x).
    scale=10**6
    return F(isqrt((x.numerator*scale**2)//x.denominator)+1,scale)


def main():
    rng=random.Random(20261124); count=accepted=ties=0
    for t in range(6000):
        d=rng.randrange(1,6); B=rng.randrange(1,6); n=rng.randrange(1,9)
        truth=[F(0)]*d; pred=[F(0)]*d; total_bound=F(0)
        for b in range(B):
            low=F(rng.randrange(1,9),8); high=low+F(rng.randrange(0,9),8)
            x=[low+(high-low)*F(rng.randrange(0,9),8) for _ in range(n)]
            V=[[F(rng.randrange(-8,9),4) for _ in range(d)] for _ in range(n)]
            mu=[sum(v[j] for v in V)/n for j in range(d)]
            y=[sum(x[i]*V[i][j] for i in range(n)) for j in range(d)]
            rms2=sum(norm2([v[j]-mu[j] for j in range(d)]) for v in V)
            delta=(high-low)/2; mid=(high+low)/2
            y0=[n*mid*m for m in mu]
            e0=n*delta**2*(n*norm2(mu)+rms2)
            avg=sum(x)/n
            y1=[n*avg*m for m in mu]
            e1=sum((z-avg)**2 for z in x)*rms2
            assert norm2([a-b for a,b in zip(y,y0)])<=e0
            assert norm2([a-b for a,b in zip(y,y1)])<=e1
            stage=rng.randrange(3)
            yy,ee=[(y0,e0),(y1,e1),(y,F(0))][stage]
            truth=[a+b for a,b in zip(truth,y)]
            pred=[a+b for a,b in zip(pred,yy)]
            total_bound+=upper_sqrt(ee); count+=2
        W=[[F(rng.randrange(-8,9),4) for _ in range(d)] for _ in range(5)]
        if t%5==0: W=[W[0]]*5; ties+=1
        exact=[dot(truth,w) for w in W]; approx=[dot(pred,w) for w in W]
        win=max(range(5),key=lambda j:approx[j]); ok=True
        for j in range(5):
            if j==win: continue
            margin=approx[win]-approx[j]
            a=[u-v for u,v in zip(W[win],W[j])]
            ok &= margin>0 and margin**2>total_bound**2*norm2(a)
        if ok:
            accepted+=1
            assert all(exact[win]>exact[j] for j in range(5) if j!=win)
    print(json.dumps(dict(cases=6000,local_bound_checks=count,accepted=accepted,forced_ties=ties,violations=0),indent=2))

if __name__=='__main__': main()
