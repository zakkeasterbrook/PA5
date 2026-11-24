"""Exact two-checkpoint adaptive-compute barrier and independent LP checks.
Data: two selected METR Earthborne Rangers mean curves, interpreted as
monotone scores on the 21-objective grid, without asserting empirical paths.
"""
from fractions import Fraction as Q
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import csv,json,random

ROOT=Path(__file__).resolve().parent

def grid(K):
    return [Q(i,K) for i in range(K+1)]

def neighbors(z,K):
    j=(K*z).numerator//(K*z).denominator
    u=Q(j,K);v=min(Q(1),u+Q(1,K))
    return u,v

def rounded(z,K):
    u,v=neighbors(z,K)
    if u==v or z==u:return [(u,Q(1))]
    return [(u,(v-z)/(v-u)),(v,(z-u)/(v-u))]

def moment_upper(mu,nu,K):
    if not nu:return Q(0)
    z=mu/nu;u,v=neighbors(z,K)
    return nu*(z*(1-z)-(z-u)*(v-z))

def check_law(P,mu,nu,K):
    assert all(p>=0 and 0<=x<=y<=1 and (x*K).denominator==(y*K).denominator==1 for x,y,p in P)
    assert sum(p for x,y,p in P)==1
    assert sum(x*p for x,y,p in P)==mu
    assert sum(y*p for x,y,p in P)==nu

def moment(P):return sum(p*x*(y-x) for x,y,p in P)

def upper_witness(mu,nu,K):
    if not nu:return [(Q(0),Q(0),Q(1))]
    return [(Q(0),Q(0),1-nu)]+[(x,Q(1),nu*p) for x,p in rounded(mu/nu,K)]

def zero_witness(mu,nu):
    return [(Q(0),Q(1),nu-mu),(Q(1),Q(1),mu),(Q(0),Q(0),1-nu)]

def no_gain_witness(mu,nu,K):
    M=moment_upper(mu,nu,K);d=nu-mu
    assert d>0 and M>=mu*d
    if not M:return zero_witness(mu,nu)
    t=mu*d/M
    return [(x,y,t*p) for x,y,p in upper_witness(mu,nu,K)]+[(x,y,(1-t)*p) for x,y,p in zero_witness(mu,nu)]

def improvement(mu,nu,K,eps):
    d=nu-mu;m=moment_upper(mu,nu,K)/d;alpha=eps/d
    assert m<mu and 0<alpha<1
    beta=min(alpha/m if m else Q(1), (1-alpha)/(1-m))
    f=[alpha+beta*(x-m) for x in grid(K)]
    return f,alpha+beta*(mu-m)

def envelope(values,z,K,maximize=False):
    """Extremum E[v(X)] over grid X of specified mean, via support size <=2."""
    x=grid(K);candidates=[]
    for i,u in enumerate(x):
        if u==z:candidates.append(values[i])
        for j in range(i+1,len(x)):
            v=x[j]
            if u<z<v:candidates.append(((v-z)*values[i]+(z-u)*values[j])/(v-u))
    return (max if maximize else min)(candidates)

def worst_save(f,mu,K):return envelope(f,mu,K)

def worst_loss(f,mu,nu,K):
    if nu==0:return Q(0)
    return nu*envelope([(1-x)*fx for x,fx in zip(grid(K),f)],mu/nu,K,True)

def exact_tests():
    rng=random.Random(92611);counts=defaultdict(int)
    for K in range(1,13):
        for j in range(1,33):
            nu=Q(j,32)
            for i in range(j):
                mu=Q(i,32);d=nu-mu;M=moment_upper(mu,nu,K)
                P=upper_witness(mu,nu,K);check_law(P,mu,nu,K);assert moment(P)==M
                P0=zero_witness(mu,nu);check_law(P0,mu,nu,K);assert moment(P0)==0
                counts['exact_moment_witness_cases']+=1
                z=mu/nu;u,v=neighbors(z,K)
                assert (z-u)*(v-z)<=z*(1-z)/K
                if nu<=1-Q(1,K):assert M>=mu*d
                if M>=mu*d:
                    P=no_gain_witness(mu,nu,K);check_law(P,mu,nu,K);assert moment(P)==mu*d
                    # This gain-tilted probability distribution has the required first mean.
                    assert sum(p*(y-x) for x,y,p in P)/d==1
                    assert sum(x*p*(y-x) for x,y,p in P)/d==mu
                    counts['exact_no_gain_certificates']+=1
                    for _ in range(2):
                        f=[Q(rng.randrange(17),16) for x in grid(K)]
                        s=worst_save(f,mu,K);loss=worst_loss(f,mu,nu,K)
                        witness_loss=sum(p*(y-x)*f[int(x*K)] for x,y,p in P)
                        assert loss>=witness_loss>=d*s
                        counts['exact_arbitrary_policy_checks']+=1
                else:
                    for a in (Q(1,100),Q(1,2),Q(99,100)):
                        eps=a*d;f,s=improvement(mu,nu,K,eps)
                        assert all(0<=x<=1 for x in f)
                        assert worst_save(f,mu,K)==s>a
                        assert worst_loss(f,mu,nu,K)==eps
                        counts['exact_strict_improvement_certificates']+=1
    return dict(counts)

def lp_opt(mu,nu,K,eps):
    """Independent LP dual formulation for all policies f(x), not restricted to affine.
    Objective max a+b mu; a+b x <= f(x) for savings; p+q x+r y >= f(x)(y-x)
    for all x<=y; p+q mu+r nu <= eps. Floating solve, separate from exact proof.
    """
    from scipy.optimize import linprog
    import numpy as np
    mu,nu,eps=map(float,(mu,nu,eps));x=np.linspace(0,1,K+1);n=K+1
    rows=[];rhs=[]
    for i,u in enumerate(x):
        row=np.zeros(n+5);row[i]=-1;row[n:n+2]=[1,u];rows.append(row);rhs.append(0)
        for v in x[i:]:
            row=np.zeros(n+5);row[i]=v-u;row[n+2:]=[-1,-u,-v];rows.append(row);rhs.append(0)
    row=np.zeros(n+5);row[n+2:]=[1,mu,nu];rows.append(row);rhs.append(eps)
    obj=np.zeros(n+5);obj[n:n+2]=[-1,-mu]
    z=linprog(obj,A_ub=rows,b_ub=rhs,bounds=[(0,1)]*n+[(None,None)]*5,method='highs')
    assert z.success,z.message
    return -z.fun

def independent_tests():
    import numpy as np
    from scipy.optimize import linprog
    rng=random.Random(41121);counts=defaultdict(int);maxerr=0;min_positive=1
    # Independent optimization over joint laws for random f, and full policy LP.
    cases=[]
    for K in [1,2,3,4,7,12,21]:
        for _ in range(45):
            j=rng.randrange(1,65);i=rng.randrange(j)
            cases.append((Q(i,64),Q(j,64),K))
        for _ in range(15):
            mu=Q(rng.randrange(1,64),64);cases.append((mu,Q(1),K))
    for mu,nu,K in cases:
        x=grid(K);pairs=[(u,v) for u in x for v in x if u<=v];d=nu-mu
        A=np.array([[1]*len(pairs),[float(u) for u,v in pairs],[float(v) for u,v in pairs]])
        b=[1,float(mu),float(nu)];f=[Q(rng.randrange(101),100) for u in x]
        for values,target in [([u*(v-u) for u,v in pairs],moment_upper(mu,nu,K)),([f[int(u*K)]*(v-u) for u,v in pairs],worst_loss(f,mu,nu,K))]:
            z=linprog([-float(v) for v in values],A_eq=A,b_eq=b,bounds=(0,None),method='highs')
            assert z.success
            err=abs(-z.fun-float(target));maxerr=max(maxerr,err);assert err<1e-8
            counts['independent_joint_law_LP_checks']+=1
        eps=d/2;opt=lp_opt(mu,nu,K,eps);M=moment_upper(mu,nu,K)
        if M>=mu*d:
            assert abs(opt-0.5)<1e-8,(mu,nu,K,opt,M)
            counts['independent_no_gain_policy_LP_checks']+=1
        else:
            f,s=improvement(mu,nu,K,eps)
            assert opt>=float(s)-1e-8 and s>Q(1,2),(mu,nu,K,opt,s)
            min_positive=min(min_positive,float(s-Q(1,2)))
            counts['independent_strict_gain_policy_LP_checks']+=1
    return dict(counts,max_joint_LP_absolute_error=maxerr,min_certified_positive_gap=min_positive)

def study_data():
    groups=defaultdict(list)
    with (ROOT/'metr_points.csv').open() as f:
        for r in csv.DictReader(f):
            if r['family']=='Earthborne Rangers':groups[r['model']].append((Q(r['cost_usd']),Q(r['mean_best_score'])))
    result=[]
    for model,rows in groups.items():
        n=len(rows);total=flat=barrier=positive=0
        for (c,mu),(b,nu) in combinations(rows,2):
            assert c<b and mu<=nu
            total+=1
            if mu==nu:flat+=1;continue
            if moment_upper(mu,nu,21)>=mu*(nu-mu):barrier+=1
            else:positive+=1
        result.append(dict(model=model,checkpoints=n,pairs=total,flat_pairs=flat,certified_no_gain_pairs=barrier,strict_gain_pairs=positive,final_mean=str(rows[-1][1]),first_cost=str(rows[0][0]),last_cost=str(rows[-1][0])))
    return result

def main():
    exact=exact_tests();print('exact',exact,flush=True)
    independent=independent_tests();print('independent',independent,flush=True)
    data=study_data();print('data',data,flush=True)
    # A worked 21-objective near-saturation case.
    mu=Q(41,42);nu=Q(1);eps=(nu-mu)/2;f,s=improvement(mu,nu,21,eps)
    example=dict(K=21,mu=str(mu),nu=str(nu),eps=str(eps),fixed_saved_fraction='1/2',certified_adaptive_saved_fraction=str(s),worst_loss=str(worst_loss(f,mu,nu,21)),numerical_optimum=lp_opt(mu,nu,21,eps))
    # A binary-valued high-potential case: analytic policy can stop on success with zero score loss;
    # for positive loss tolerance use the exact affine witness.
    mu=Q(3,5);nu=Q(9,10);eps=Q(3,100);f,s=improvement(mu,nu,1,eps)
    example2=dict(K=1,mu=str(mu),nu=str(nu),eps=str(eps),fixed_saved_fraction=str(eps/(nu-mu)),certified_adaptive_saved_fraction=str(s),worst_loss=str(worst_loss(f,mu,nu,1)),numerical_optimum=lp_opt(mu,nu,1,eps))
    result=dict(exact=exact,independent=independent,data=data,examples=[example,example2]);print(json.dumps(result,indent=2));(ROOT/'barrier_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
