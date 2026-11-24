"""Check the archived multistage candidates by exact rational Bellman bounds.
This verifies feasibility, not global optimality of the multistage problem.
"""
import csv,json
from collections import defaultdict
from fractions import Fraction as Q
from pathlib import Path
from robust import exact_bound,fixed,uniform
ROOT=Path(__file__).resolve().parent

def verify():
    groups=defaultdict(list)
    with (ROOT/'metr_points.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['family']=='Earthborne Rangers':
                groups[r['model']].append((Q(r['cost_usd']),Q(r['mean_best_score'])))
    records=json.loads((ROOT/'results.json').read_text())
    out=[]
    for r in records:
        costs,means=map(list,zip(*groups[r['model']]))
        cap=costs[-1]; normalized=[c/cap for c in costs];K=21
        F=[[Q(x) for x in row] for row in r['F']]
        a=list(map(Q,r['a']));b=list(map(Q,r['b']));eps=Q(r['epsilon'])
        assert len(F)==len(costs)-1 and all(len(row)==K+1 for row in F)
        assert len(a)==len(b)==len(costs)
        assert all(0<=x<=1 for row in F for x in row)
        assert all(x<=y for row in F for x,y in zip(row,row[1:]))
        assert all(x<=y for row,nextrow in zip(F,F[1:]) for x,y in zip(row,nextrow))
        save=exact_bound(F,a,normalized,means,K,'save')
        loss=exact_bound(F,b,normalized,means,K,'loss')
        assert save==Q(r['certified_savings'])
        assert loss==Q(r['certified_loss'])<=eps
        assert cap*(1-save)==Q(r['cap'])
        fixed_cap,_=fixed(costs,means,eps)
        assert fixed_cap==Q(r['fixed_cap'])
        assert uniform(costs,means,K)==(Q(r['uniform_savings']),Q(r['uniform_worst_loss']))
        out.append(dict(model=r['model'],epsilon=str(eps),certified_cap=str(cap*(1-save)),certified_loss=str(loss),fixed_cap=str(fixed_cap),status='pass'))
    assert len(out)==2
    print(json.dumps({'archived_multistage_certificates':out},indent=2))
if __name__=='__main__':verify()
