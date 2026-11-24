"""Dual KV block retrieval for a bias-free linear attention readout.
Real-arithmetic certificate; floating-point evaluation is not rigorous.
"""
import numpy as np


def dense(K,V,q,W):
    s=K@q
    return int(np.argmax((np.exp(s-s.max())@V)@W))


class Cache:
    def __init__(self,K,V,b=128):
        K,V=np.asarray(K,dtype=np.float64),np.asarray(V,dtype=np.float64)
        if K.ndim!=2 or V.ndim!=2 or len(K)!=len(V) or not len(K) or b<1 or len(K)%b:
            raise ValueError('Require full equal-size blocks')
        if not np.isfinite(K).all() or not np.isfinite(V).all():
            raise ValueError('Nonfinite cache')
        self.K,self.V,self.b=K.reshape(-1,b,K.shape[1]),V.reshape(-1,b,V.shape[1]),b
        self.kc=self.K.mean(axis=1)
        self.kr=np.linalg.norm(self.K-self.kc[:,None,:],axis=2).max(axis=1)
        self.vm=self.V.mean(axis=1)
        self.vr=np.sqrt(np.sum((self.V-self.vm[:,None,:])**2,axis=(1,2)))
        self.raw=K.nbytes+V.nbytes
        self.meta=sum(x.nbytes for x in (self.kc,self.kr,self.vm,self.vr))

    def decide(self,q,W,mode='dual'):
        """q includes attention scaling; W is bias-free. Count repeated fallback reads."""
        if mode not in ('dual','values'): raise ValueError('Unknown mode')
        q,W=np.asarray(q,dtype=np.float64),np.asarray(W,dtype=np.float64)
        if not np.isfinite(q).all() or not np.isfinite(W).all() or W.ndim!=2 or W.shape[1]<2:
            raise ValueError('Require finite query and >=2 outputs')
        B,b=len(self.K),self.b
        K,V=self.K,self.V
        state=np.zeros(B,dtype=np.int8)
        reads,nk,nv=self.meta,0,0
        def fallback():
            return dict(token=dense(K.reshape(-1,K.shape[-1]),V.reshape(-1,V.shape[-1]),q,W),accepted=False,bytes=reads+self.raw,keys=nk+B,values=nv+B,bound=None)
        c=self.kc@q
        r=self.kr*np.linalg.norm(q)
        shift=np.max(c+r)
        if not np.isfinite(shift) or shift-np.max(c-r)>600 or np.max(np.abs(c)+r)>1e4:
            return fallback()
        lo,hi=np.exp(c-r-shift),np.exp(c+r-shift)
        mid,delta=(lo+hi)/2,(hi-lo)/2
        y=b*mid[:,None]*self.vm
        err=b*delta*np.hypot(np.linalg.norm(self.vm,axis=1),self.vr/np.sqrt(b))
        weights=np.empty((B,b),dtype=np.float64)
        def keys(ids):
            nonlocal reads,nk
            if not len(ids): return
            x=np.exp(K[ids]@q-shift)
            weights[ids]=x
            m=x.mean(axis=1)
            y[ids]=(b*m)[:,None]*self.vm[ids]
            err[ids]=np.linalg.norm(x-m[:,None],axis=1)*self.vr[ids]
            state[ids]=1
            reads+=len(ids)*K[0].nbytes
            nk+=len(ids)
        def values(ids):
            nonlocal reads,nv
            if not len(ids): return
            y[ids]=np.einsum('ij,ijk->ik',weights[ids],V[ids])
            err[ids]=0
            state[ids]=2
            reads+=len(ids)*V[0].nbytes
            nv+=len(ids)
        if mode=='values': keys(np.arange(B))
        for step in range(6):
            total=y.sum(axis=0)
            E=float(err.sum())
            scores=total@W
            win=int(np.argmax(scores))
            norms=np.linalg.norm(W[:,win,None]-W,axis=0)
            margins=scores[win]-scores
            # Numerical cushion, not directed rounding.
            scale=np.sum(np.linalg.norm(y,axis=1))+E
            guard=1e-10*max(scale,1e-300)*max(np.linalg.norm(W,axis=0).max(),1e-300)
            ok=margins>E*norms+guard
            ok[win]=True
            if np.isfinite(scores).all() and np.isfinite(E) and ok.all():
                return dict(token=win,accepted=True,bytes=reads,keys=nk,values=nv,bound=E)
            if step==5 or np.all(state==2): break
            ids=np.flatnonzero(state<2)
            take=min(len(ids),max(1,B//(16 if step==0 else 8 if step==1 else 4)))
            ids=ids[np.argsort(-err[ids],kind='stable')[:take]]
            fresh,old=ids[state[ids]==0],ids[state[ids]==1]
            keys(fresh); values(old)
        return fallback()
