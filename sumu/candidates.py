import time
import numpy as np
from .aps import aps
from .utils.math_utils import subsets
from statistics import mean
import pandas as pd
from scipy.stats import multivariate_normal

def rnd(K, *, data, **kwargs):

    n = data.n
    C = dict()
    for v in range(n):
        C[v] = tuple(
            sorted(
                np.random.choice(
                    [u for u in range(n) if u != v], K, replace=False
                )
            )
        )
    return C, None


def opt(K, **kwargs):

    scores = kwargs.get("scores")
    n = kwargs.get("n")

    C = np.array(
        [[v for v in range(n) if v != u] for u in range(n)], dtype=np.int32
    )
    pset_posteriors = aps(
        scores.all_candidate_restricted_scores(C), as_dict=True, normalize=True
    )

    C = dict()
    for v in pset_posteriors:
        postsums = dict()
        for candidate_set in subsets(
            set(pset_posteriors).difference({v}), K, K
        ):
            postsums[candidate_set] = np.logaddexp.reduce(
                [
                    pset_posteriors[v][pset]
                    for pset in subsets(candidate_set, 0, K)
                ]
            )
        C[v] = max(postsums, key=lambda candidate_set: postsums[candidate_set])
    return C, None


def greedy(
    K,
    *,
    scores,
    params={"k": 6, "t_budget": None, "criterion": "score", "opt_criterion": "max", "discount": "none", "d": 0, "var": -1},
    **kwargs,
):
    data = kwargs.get("data")
   # print(data.n)
   # print(data.discrete)
   # print(data.arities)
    logN = np.log(data.N)
    arities = data.arities;

    k = params.get("k")
    if k is not None:
        k = min(k, K)

    #print("k:" + str(k))
    t_budget = params.get("t_budget")
    criterion = params.get("criterion")
    #print("criterion:",criterion);
    opt_criterion = params.get("opt_criterion")
    #print("opt_criterion:",opt_criterion);
    discount = params.get("discount")
    #print("discount:",discount)
    d = params.get("d")

    var = params.get("var")
    #print("d:" +  str(d));

    assert criterion in ("score", "gain", "random", "lgain", "lscore", "lrandom")
    assert opt_criterion in ("max", "min", "random", "mean")
    assert discount in ("none","bic")


    lcache = {frozenset(): 0} #works for all sets
    #print(data.data)

  #  print(data.data.shape)

    #df = pd.read_csv('/Users/ajhyttin/sumu_journal/R/data/diabetes_400_1.csv', header=None, sep= " ",dtype='category')
    df=pd.DataFrame(data=data.data)

    #print(df)

    #print(data.discrete)

    def loglikelihood4set(s):
       ss = frozenset(s)
       try:
          return lcache[ss]
       except KeyError:
          score = 0;
          if ( data.discrete ):
             ctab=df[np.array(s)].value_counts()
             score = (ctab*(np.log(ctab)-np.log(ctab.sum()))).sum()
          else:
             #print(s);
             mean = df[np.array(s)].mean(axis=0); #print(mean);
             cov = df[np.array(s)].cov(ddof=0); #print(cov);
             score =  multivariate_normal.logpdf(df[np.array(s)],mean=mean,cov=cov).sum()
          lcache[ss] = score
          return score

    def loglikelihood_localscore(node,given):
       #print("node:");print(node)
       #print("given:");print(given)
       score=loglikelihood4set((node,)+given)-loglikelihood4set(given)
       #print(score)
       return score


    #print(loglikelihood4set( (0,) ))
    #print(loglikelihood4set( (0,1) ))
    #print(loglikelihood4set(range(data.n)))
   # print(loglikelihood_localscore(2,()))
   # print(loglikelihood_localscore(2,(7)))
   # print(loglikelihood_localscore(2,np.array([])) )
   # print(loglikelihood_localscore(2,np.array([7])) )
    #print('plain calls done.')
    #assert( data.discrete )

    #local_score = lambda v, P: scores._local(v,np.array(P)) + penalty(v,P) if discount == "bic" else scores._local(v,np.array(P))

    local_score = lambda v, P, criterion: loglikelihood_localscore(v,P) if (criterion == "lgain" or criterion == "lscore") else scores._local(v,np.array(P))

    goodness_given_set = lambda v, S, u, criterion: local_score(
        v, S + (u,),criterion ) - local_score(v, S,criterion) if (criterion == "gain" or criterion == "lgain" ) else local_score(v, S + (u,),criterion)

    goodnesses = lambda v, u, criterion: {
                    goodness_given_set(v, S, u, criterion)
                    for S in subsets(
                        C[v],
                        0,
                        [
                            len(C[v])
                            if scores.maxid == -1
                            else min(len(C[v]), scores.maxid - 1)
                        ][0],
                    )
                }.union( {
                    goodness_given_set(v, S, u, criterion)
                    for S in subsets(
                        set(range(data.n)).difference({v}),
                        0,
                        d,
                    )
                })

    goodness = lambda v, u, criterion, opt_criterion: max(
                    goodnesses( v, u, criterion)
                ) if opt_criterion == "max" else min(
                    goodnesses( v, u, criterion)
                ) if (opt_criterion == "min" ) else mean(
                    goodnesses( v, u, criterion)
                )


    def k_highest_uncovered(v, U, k, criterion, opt_criterion):
        uncovereds = {
            u: max(
                {
                    goodness(v, u, criterion, opt_criterion)
                }
            )
            for u in U
        }

        k_highest = list()
        while len(k_highest) < k:
            u_hat = max(uncovereds, key=lambda u: uncovereds[u])
            k_highest.append(u_hat)
            del uncovereds[u_hat]
        return k_highest

    def get_k(t_budget):
        if t_budget < 0:
            return K
        U = set(range(1, len(C)))
        t_used = list()
        t = time.time()
        t_pred_next_add = 0
        i = 0
        u_hat = []
        while len(C[0]) < K and sum(t_used) + t_pred_next_add < t_budget:
            criterion_local = np.random.choice(["score","gain"]) if criterion == "random" else np.random.choice(["lscore","lgain"]) if criterion == "lrandom" else criterion;
            opt_criterion_local = np.random.choice(["max","min","mean"]) if opt_criterion == "random" else opt_criterion;
            u_hat = k_highest_uncovered(0, U, 1, criterion_local, opt_criterion_local)
            C[0] += u_hat
            U -= set(u_hat)
            t_prev = time.time() - t - sum(t_used)
            t_used.append(t_prev)
            t_pred_next_add = 2 * t_prev
            i += 1
        C[0] = [v for v in C[0] if v not in u_hat] #error uhat no t defined here
        U.update(set(u_hat))
        k = K - len(C[0])
        criterion_local = np.random.choice(["score","gain"]) if criterion == "random" else np.random.choice(["lscore","lgain"]) if criterion == "lrandom" else criterion;
        opt_criterion_local = np.random.choice(["max","min","mean"]) if opt_criterion == "random" else opt_criterion;
        C[0] += k_highest_uncovered(0, U, k, criterion_local, opt_criterion_local)
        C[0] = tuple(C[0])
        scores.clear_cache()
        return k

    C = dict({int(v): list() for v in range(scores.data.n)})

    if t_budget is not None:
        t_budget /= scores.data.n
        k = get_k(t_budget)

    vrange = [v for v in C if len(C[v]) == 0]
    #if ( var >= 0 ):
    #    vrange = [var]
    for v in vrange:
        if ( var >= 0 and v != var):
           for kk in range(K):
               C[v] +=[0];
           continue
        U = [u for u in C if u != v]
        while len(C[v]) < K - k:
            # We have to set criterions at this level so that they are the same for call nodes considered as candidates
            criterion_local = np.random.choice(["score","gain"]) if criterion == "random" else np.random.choice(["lscore","lgain"]) if criterion == "lrandom" else criterion;
            opt_criterion_local = np.random.choice(["max","min","mean"]) if opt_criterion == "random" else opt_criterion;
            u_hat = k_highest_uncovered(v, U, 1, criterion_local, opt_criterion_local)
            #print( u_hat );
            C[v] += u_hat
            #assert(len(C[v]) < 2);
            U = [u for u in U if u not in u_hat]

        criterion_local = np.random.choice(["score","gain"]) if criterion == "random" else np.random.choice(["lscore","lgain"]) if criterion == "lrandom" else criterion;
        opt_criterion_local = np.random.choice(["max","min","mean"]) if opt_criterion == "random" else opt_criterion;
        C[v] += k_highest_uncovered(v, U, k, criterion_local, opt_criterion_local)
        C[v] = tuple(C[v])
        scores.clear_cache()
        lcache = {frozenset(): 0}
    return C, {"k": k}


candidate_parent_algorithm = {"opt": opt, "rnd": rnd, "greedy": greedy}
