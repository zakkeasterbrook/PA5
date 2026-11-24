# The Adaptive Compute Barrier

**Zakk Easterbrook | Preliminary AI-assisted research note | November 24, 2026 | Version 0.1.0**

## Status and scope

This note derives a sharp two-checkpoint distributionally robust stopping theorem, an exact two-variable optimizer, and a polynomial-size extension to a restricted multi-checkpoint policy class. The proofs concern a precisely specified mathematical model. The literature search is preliminary; scientific priority, peer review, and deployment-scale speed or billing savings have not been established. No new model inference was purchased. This public version is a research-software note, not a peer-reviewed paper.

## 1. Problem

Let X and Y be the retained, verifiable best scores at two fixed spending checkpoints c < C, with X <= Y almost surely. Scores lie on G = {0, 1/K, ..., 1}, and only the population means mu = E[X], nu = E[Y] are known. Write d = nu - mu > 0 and consider a permissible worst-case expected normalized score loss 0 < epsilon < d. Every law on ordered pairs in G with these two means is allowed; there are no independence assumptions. The checkpoint costs are modeled caps, not random invoices. The policy has no other observations or prior task-specific features. It stops at the early checkpoint with probability f(X), where f: G -> [0,1], using independent private randomization, and otherwise continues to C. Define

    s(f) = inf_P E_P[f(X)],
    L(f) = sup_P E_P[f(X)(Y-X)].

The robust problem is to maximize s(f) subject to L(f) <= epsilon. The dollar guarantee is at least (C-c)s(f) in expected cap savings, or expected cap at most C-(C-c)s(f). These extrema may come from different admissible laws. A score-independent randomized budget achieves s = alpha = epsilon/d and loss epsilon for every law, and no score-independent policy does better.

If d=0, nondecreasing scores are equal almost surely, so stopping immediately incurs zero loss. If epsilon >= d, stopping immediately is feasible. The strict-benefit characterization below concerns the interior tolerance 0 < epsilon < d; zero loss has separate boundary cases.

## 2. Sharp moment lemma and exact robust policy evaluation

For any f, every ordered pair (x,y) with x<1 can be replaced by a randomized pair (x,1) with probability (y-x)/(1-x), and (x,x) otherwise. This preserves both means and the expected payoff f(X)(Y-X). In the no-gain part, replace each (x,x) by (1,1) with probability x and (0,0) otherwise. This too preserves both means and payoff. For x=1, no transformation is necessary. Thus every payoff achievable by a general law is achievable by a law supported on Y=0,X=0 and Y=1,X in G. Since E[Y]=nu, conditional X given Y=1 has mean z=mu/nu. Conversely any such conditional distribution gives an admissible joint law. Consequently

    L(f) = nu * CAV_G[(1-x)f(x)](z),

where CAV_G denotes the upper concave envelope, equivalently the maximum expected value over grid distributions with the specified mean. Likewise

    s(f) = VEX_G[f](mu),

where VEX_G denotes the lower convex envelope. Every first-score distribution of mean mu can be extended to the specified second mean: conditionally set Y=1 with probability d/(1-mu), and Y=X otherwise. Finite moment linear programs attain both envelopes using at most two support points.

For the moment M = sup_P E[X(Y-X)], choose f(x)=x. Let a,b be adjacent grid values bracketing z (if z itself is a grid point, either adjacent bracket containing it suffices). On the grid (x-a)(x-b) >= 0, so E[X^2] >= (a+b)z-ab = z^2+(z-a)(b-z) conditionally on Y=1. Equality is attained by the two-point distribution on a,b with mean z. Thus, exactly,

    V = (z-a)(b-z),
    M = nu [z(1-z)-V].

The minimum possible moment is zero, attained by assigning mass d at (0,1), mu at (1,1), and 1-nu at (0,0). Mixtures of these two extremal laws attain every intermediate moment. These arguments also work on any finite score grid containing 0 and 1, with a,b the bracketing grid values.

## 3. Theorem: a necessary and sufficient barrier

**Strict robust savings improvement over the best independent budget exists if and only if**

    V > z(1-z)(1-nu).

Equivalently M < mu*d. Proof:

### No-gain direction

If M >= mu*d, mix the upper- and zero-moment witnesses to obtain an admissible law P* for which E[X(Y-X)]=mu*d (when M=0 use the zero witness). Define the gain-tilted probability law on grid scores

    Q(x) = E_P*[(Y-X) 1{X=x}]/d.

Its total mass is one and its mean is mu. It is therefore a feasible first-score marginal. For any randomized stopping rule, including nonmonotone ones,

    L(f) >= E_P*[f(X)(Y-X)] = d E_Q[f(X)] >= d s(f).

If L(f)<=epsilon, then s(f)<=epsilon/d. The constant policy f=alpha attains equality. This is an exact minimax impossibility result, not a claim about the best policy for a known joint law.

### Strict-gain direction

If M < mu*d, put m=M/d < mu. Let alpha=epsilon/d and choose beta>0 sufficiently small that f(x)=alpha+beta(x-m) is in [0,1] on the grid. For example choose beta <= min(alpha/m, (1-alpha)/(1-m)), ignoring the first term if m=0. Then every law gives

    E[f(X)] = alpha + beta(mu-m) > alpha,
    sup_P E[f(X)(Y-X)] = alpha*d + beta(M-m*d) = epsilon.

Hence a strictly better robust policy exists. Combining both directions and substituting the exact formula for M proves the criterion.

### Simple universal corollary

For a finite grid with maximum gap h, write A=z-a and B=b-z. Since A<=z and B<=1-z, the function AB/(A+B) (increasing in either argument on the nonnegative domain) is at most z(1-z). Thus V<=h*z(1-z). If nu<=1-h, the strict-benefit condition cannot hold. For the uniform K-subdivision grid, h=1/K. The threshold is sufficient for no gain, not a necessary boundary by itself; the full criterion is exact. For K=21 the sufficient no-gain threshold is 20/21, about 95.238%. For continuous scores, a conditional point mass at z makes V=0, so the no-gain direction holds for all means under the analogous ambiguity class.

## 4. Exact two-variable policy optimizer

For any f with guaranteed savings s, the lower convex envelope at mu admits an affine supporting minorant ell(x)=a+b*x with ell(mu)=s and ell(x)<=f(x) on the grid. The rule f'(x)=max(0,ell(x)) is pointwise no greater than f; it cannot increase worst loss, and every admissible first marginal gives E[f'(X)] >= ell(mu)=s. Thus there is no loss of optimality in optimizing this clipped-affine family (with objective ell(mu)). A decreasing clipped-affine rule cannot beat alpha: the admissible zero-moment law concentrates all gains at X=0, so L(f') >= d*f'(0) >= d*s(f'). Therefore b>=0 can be assumed for finding any strict improvement. The constant rule is always available.

The conditions are a+b<=1, b>=0, and

    nu * CAV_G[(1-x)max(0,a+b*x)](z) <= epsilon.

For each grid pair u<z<v, let p=(v-z)/(v-u), q=1-p, A=nu*p*(1-u), B=nu*q*(1-v). The constraint for its two-point distribution is equivalent to three nontrivial linear inequalities:

    A(a+b*u) <= epsilon,
    B(a+b*v) <= epsilon,
    (A+B)a + (A*u+B*v)b <= epsilon.

A singleton at z, if z is a grid value, supplies one analogous inequality. These are equivalent because the sum of two positive parts is the maximum over the four choices of whether to retain each argument, with the all-zero choice trivial. Thus the exact optimum is a two-variable linear program, rather than a search over arbitrary stopping rules. The implementation clips a bounded rational polygon and maximizes a+b*mu over its vertices. To bound the search region without sacrificing an optimum, restrict a+b*mu>=alpha, giving 0<=b<=(1-alpha)/(1-mu) and alpha-b_max*mu<=a<=1. This is valid since mu<1. The solver constructs the policy using Python Fraction arithmetic, not a floating-point optimizer. `lp_opt` in barrier.py independently solves the original policy problem with K+1 unrestricted policy probabilities using LP dual constraints; it is only used for a numerical cross-check. All claims of strict improvement and no improvement in the theorem are exact.

### Exact worked examples

* Binary score: mu=3/5, nu=9/10, epsilon=3/100. Stop always if X=1; stop with probability 1/10 if X=0. Robust saved fraction is 16/25 (64%) versus 1/10 (10%) for the optimal independent budget. Loss is exactly 3/100. Since only X=0 can gain, any feasible rule must have f(0)<=1/10 and f(1)<=1, proving optimality directly.
* K=21, mu=41/42, nu=1, epsilon=1/84. Stop with probability 1 at X=1, 1/2 at X=20/21, 0 otherwise. The exact optimizer achieves 3/4 (75%) saved fraction vs 1/2 (50%) fixed. This is exact under the ambiguity class. For example, half the population at each of the two top grid scores gives the upper bound of 3/4, since loss restricts f(20/21)<=1/2. These are constructed mathematical examples, not measured AI workloads.

## 5. Multi-checkpoint extension

For T costs c_i and a monotone score path x_i, take a randomized monotone stopping surface F_i(x), nondecreasing in checkpoint and score, 0<=F<=1; use a single independent uniform U in [0,1) and stop when U<F_i(x_i), or at the final checkpoint. The pathwise conditional expected savings in dollars and score loss are exactly

    S_F(x) = sum_{i<T} (c_{i+1}-c_i) F_i(x_i),
    L_F(x) = sum_{i<T} (x_{i+1}-x_i) F_i(x_i).

This class includes all independent randomized checkpoint caps, but not every possible history-dependent stopping rule. Under only the marginal mean constraints, the worst expectation of each additive quantity is a finite moment LP over monotone paths. Its dual seeks affine functions of the entire path's scores, with an intercept and one coefficient per checkpoint, below the savings functional or above the loss functional for all paths. All inequalities over paths can be represented by Bellman potentials on the time-score DAG: each transition connects j to k>=j, with local rewards given by the two terms above minus the appropriate dual coefficient times j/K. This yields O(T*K^2) constraints and O(T*K) variables (plus the mean multipliers). The lower and upper Bellman recurrences use min and max respectively. Linear programming jointly optimizes F and the dual certificates. In robust.py, a floating LP discovers candidates; we rationalize the surfaces and coefficients and use exact Fraction Bellman recurrences to certify their worst-case expected cap upper bound and loss upper bound. If the loss bound exceeds epsilon because of rounding, scale F and coefficients by epsilon/loss (mixing with the never-stop policy). This provides an exact feasible-policy guarantee, not a certified numerical upper bound on the optimal savings of the entire class. The numerical optimum is separately labeled. No global no-improvement claim for all history-dependent multistage policies is made.

For the uniform random target surface F_i(x)=x, expected cap savings are exactly the area sum_i (c_{i+1}-c_i)*mu_i. Its worst-case loss can also be calculated exactly. Let D_i be integer score-count increments, including initial D_1=K*x_1. Conditional expected loss is sum_{i<j}D_i D_j/K^2. For any path with total count S>0, replace it with zero with probability 1-S/K and with a full-count path D+(K-S)e_J otherwise, where P(J=j)=D_j/S. This preserves every expected increment and increases or preserves every expected cross product D_iD_j, by factor (2K-S)/K. Thus a worst-case distribution may be supported on zero or fully saturated terminal count K. Conditional on full, the increments have means a_i=K*(mu_i-mu_{i-1})/mu_T with mu_0=0. For full paths the cross-product sum is (K^2-sum D_i^2)/2. The minimum squared sum is attained by rounding each a_i to its adjacent integers, with sum K: take differences of floor(cumulative a_i+U) for uniform U. Writing phi(a)=floor(a)^2+(2floor(a)+1)(a-floor(a)), the sharp worst-case expected score loss is mu_T*(K^2-sum phi(a_i))/(2K^2), or zero if mu_T=0. This auxiliary formula is implemented in robust.py; the two-checkpoint theorem and optimizer have the extensive independent tests reported below.

## 6. Application to recent public AI data

Original source: METR, *AI R&D-relevant model capabilities are improving rapidly and are getting harder to measure*, November 20, 2026, Appendix I (Figure 58 cost-scaling data) and Earthborne Rangers task description in Appendix N.

- https://metr.org/blog/2026-11-20-q4-2026-capability-update/
- https://metr.org/assets/images/q4-2026-capability-update/fe5f1999eeec0af5.json

The CSV is the 221 selected public values transcribed in the preceding analysis, including 123 AZsPCs values not used here. The present study analyzes only the 98 selected Earthborne Rangers checkpoints (48 Astra, 50 Sol). The source measurements are credited to METR, not to this project. The selected factual data are included for reproduction; no claim of ownership or relicensing of third-party materials is made. See DATA_SOURCES.md. The published means are rounded decimals, treated as exact inputs for a conditional calculation. We assume a common cohort, retained monotone scores on a 21-objective grid, observable and retainable verified progress, and deterministic checkpoint caps. Those assumptions must be verified with individual run data before prospective deployment. The displayed means are not population confidence bounds; rounding, verifier overhead, aggregation, interpolation, incomplete runs and actual billing may change the analysis. Costs in the source are modeled from token prices. No actual trajectory was reconstructed or inferred. No new model experiment was run.

| Model (source label) | Checkpoints | All pairs | Equal-mean pairs | Strictly increasing pairs certified no-gain | Pairs with strict gain |
|---|---:|---:|---:|---:|---:|
| GPT-6 Astra | 48 | 1,128 | 27 | 1,101 | 0 |
| GPT-5.6 Sol | 50 | 1,225 | 64 | 1,161 | 0 |
| Total | 98 | 2,353 | 91 | 2,262 | 0 |

Equal-mean pairs can stop at the earlier checkpoint without loss under the model. Each of the 2,262 strictly increasing pairs satisfies the exact barrier, for every interior loss tolerance. Both selected final means are below 20/21. The result is an identifiability limitation of aggregate evidence, not proof that actual METR agents cannot benefit from adaptation.

At a tolerance of 0.01 on the normalized score scale, the multistage monotone-surface numerical optimization did not improve meaningfully over the exact optimal independent checkpoint lottery:

| Model | Final cap | Best independent mean cap | Certified cap upper bound for optimized adaptive candidate |
|---|---:|---:|---:|
| GPT-6 Astra | $909.318 | $823.337600980049 | $823.337600980049 |
| GPT-5.6 Sol | $354.069 | $127.321076858463 | $127.321076874305 |

These are modeled checkpoint caps, not actual bills. The matching estimates do not prove exact optimum for the multistage class or every adaptive policy. See results.json for rational certificates, including monotone F, dual coefficients, exact loss and savings bounds.

## 7. Reproduction and evidence

From an isolated directory with Python and NumPy/SciPy installed, execute:

    python barrier.py
    python exact_policy.py
    python robust.py --quick

The first script executes 25,344 exact-arithmetic test cases/checks (6,336 sharp moment cases, 5,044 no-gain certificates, 10,088 arbitrary-policy checks, 3,876 strict-gain certificates). It independently checks 840 joint-law LP optima and 420 full-policy LP optima against the result. The second runs 525 additional independent exact-policy versus full-policy LP comparisons; 237 have strict benefit and 288 do not. Maximum absolute floating discrepancy in these latter comparisons: 2.22e-15. These are mathematical checks, not independent model experiments; the exact proof, not the finite testing, establishes the stated theorem. The third reruns the two selected multi-checkpoint fits with the one-point score-loss tolerance, certifying the returned policies rationally. It overwrites results.json; omit --quick to examine five tolerances. Numeric solver timings vary by hardware. The generated JSON files contain the results from the present executions.

## 8. Related work and novelty limits

Robust optimal stopping, moment linear programs, convex envelopes, LP duality, and adaptive test-time inference are established fields. See, for example, the related 2024 *Robust Multiple Stopping—A Duality Approach*, https://doi.org/10.1287/moor.2021.0237, and recent adaptive-compute work including *Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling* (November 16, 2026, https://arxiv.org/abs/2611.21241) and *Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization* (https://arxiv.org/abs/2604.14853). These papers tackle different assumptions or objectives; a preliminary search does not establish scientific priority for our exact inequality or clipped-affine reduction. The candidate research contribution here is the sharp score-grid-specific barrier, its constructive adversarial witness and strict-gain rule, the exact two-dimensional policy optimizer, and the application to published frontier-agent curves. The key practical conclusion is that marginal mean curves alone may be insufficient to certify the benefit of adaptive reasoning; paired verified progress and future gains can resolve this ambiguity. The next validation should use independently reviewed proofs, a systematic prior-art search, and run-level data to compare prospective policies under matched total cost, including the cost of verification. We do not claim a world-first result, a universal lower bound on AI computation, or a demonstrated commercial speedup.


Related prior art also includes Georghiou, Tsoukalas and Wiesemann, *On the Optimality of Affine Decision Rules in Distributionally Robust Optimization*, Management Science 72(2), 1456–1471, https://doi.org/10.1287/mnsc.2023.00053. Their broader affine-decision-rule theory reinforces the need for a narrow novelty claim, not a claim to have invented affine robust optimization.

## 9. Publication verification

Before publication, the original `barrier.py` and `exact_policy.py` were rerun on Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0. Both generated JSON result objects exactly matched the archived ones. The archived two multistage policy certificates were then checked by `verify_archived.py` in exact rational arithmetic, without relying on a fresh floating solver optimum. This checks feasible bounds, not global multistage optimality. The upstream METR report and chart JSON were verified to be accessible, and selected chart entries were spot-checked; a fresh download of the entire upstream file into the execution environment was unavailable, so a row-by-row upstream comparison is not claimed. No independent human review or research priority verification is asserted.
