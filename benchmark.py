"""Run with OPENBLAS_NUM_THREADS=1. Synthetic fixtures, not a pretrained LLM."""
import argparse,csv,time
import numpy as np
from pa5 import Cache,dense


def fixture(rng,n,b,kind,d=32,outputs=16):
    B=n//b
    q=rng.normal(size=d); q/=np.linalg.norm(q)
    centers=rng.normal(size=(B,1,d))
    noise={'clustered':.001,'concentrated':.15,'random':1.,'outlier':.001,'flat':0.,'tie':.001}[kind]
    if kind in ('random','flat'): centers*=0
    if kind=='concentrated': centers[0,0]+=18*q
    K=centers+noise*rng.normal(size=(B,b,d))
    if kind=='outlier': K[B//2,0]+=25*q
    V=rng.normal(size=(B,1,d))+.15*rng.normal(size=(B,b,d))
    if kind=='random': V=rng.normal(size=(B,b,d))
    W=rng.normal(size=(d,outputs))
    if kind=='tie': W[:]=W[:,[0]]
    return K.reshape(n,d),V.reshape(n,d),q,W


def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--out',default='trials.csv');args=p.parse_args()
    rng=np.random.default_rng(20261124); rows=[]
    cases=['clustered','concentrated','random','outlier','flat','tie']
    sizes=[4096,65536] if args.quick else [4096,65536,262144,524288]
    for n in sizes:
        for kind in cases:
            for trial in range(2 if args.quick else 4):
                K,V,q,W=fixture(rng,n,128,kind)
                t=time.perf_counter(); C=Cache(K,V); build=time.perf_counter()-t
                ref=dense(K,V,q,W)
                for mode in ('dense','dual','values'):
                    times=[]
                    for rep in range(3):
                        t=time.perf_counter()
                        if mode=='dense': z=dict(token=dense(K,V,q,W),accepted=False,bytes=C.raw,keys=len(C.K),values=len(C.V),bound=None)
                        else: z=C.decide(q,W,mode)
                        times.append(time.perf_counter()-t)
                    rows.append(dict(n=n,kind=kind,trial=trial,mode=mode,correct=int(z['token']==ref),accepted=int(z['accepted']),read_fraction=z['bytes']/C.raw,key_fraction=z['keys']/len(C.K),value_fraction=z['values']/len(C.V),ms=1000*np.median(times),build_ms=1000*build,metadata_fraction=C.meta/C.raw))
                    assert z['token']==ref,(n,kind,trial,mode,z,ref)
            print(n,kind,flush=True)
    with open(args.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__=='__main__': main()
