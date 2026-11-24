"""Additional seeded floating-point stress; not an interval arithmetic proof."""
import csv,json
import numpy as np
from pa5 import Cache,dense


def main():
    rng=np.random.default_rng(507024); rows=[]
    for i in range(2000):
        b=int(rng.choice([4,8,16,32])); B=int(rng.integers(1,17)); d=int(rng.integers(1,17)); o=int(rng.integers(2,33))
        q=rng.normal(size=d)*10**rng.uniform(-1,1)
        K=rng.normal(size=(B,1,d))+10**rng.uniform(-7,1)*rng.normal(size=(B,b,d))
        V=(rng.normal(size=(B,1,d))+rng.normal(size=(B,b,d)))*10**rng.uniform(-4,4)
        W=rng.normal(size=(d,o)); kind=['ordinary','outlier','tie','near-tie','flat'][i%5]
        if kind=='outlier': K[0,0]+=100*q
        if kind=='tie': W[:]=W[:,[0]]
        if kind=='near-tie': W[:]=W[:,[0]]+1e-12*W
        if kind=='flat': K[:]=K[0,0]
        K,V=K.reshape(B*b,d),V.reshape(B*b,d); C=Cache(K,V,b)
        ref=dense(K,V,q,W)
        for mode in ('dual','values'):
            z=C.decide(q,W,mode)
            rows.append(dict(case=i,kind=kind,mode=mode,n=B*b,d=d,outputs=o,correct=int(z['token']==ref),accepted=int(z['accepted']),read_fraction=z['bytes']/C.raw))
            assert z['token']==ref,(i,kind,mode,z,ref)
    with open('stress_trials.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(fixtures=2000,evaluations=len(rows),accepted=sum(r['accepted'] for r in rows),disagreements=sum(1-r['correct'] for r in rows)),indent=2))

if __name__=='__main__': main()
