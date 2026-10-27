"""Versioned integer-only planning; identical evidence gives identical assignments."""
from fractions import Fraction
from mtcp.kernel import digest

SETTINGS = dict(version=1, ewma_denominator=5, max_share_change_percent=10,
                minimum_improvement_percent=2, candidate_limit=10000)
COST_FIELDS = ('train_us', 'setup_us', 'audit_us', 'audit_setup_us', 'evaluation_us')


def validate_cost(cost):
    if not isinstance(cost, dict) or any(type(cost.get(k)) is not int or not 0 <= cost[k] <= 10**12 for k in COST_FIELDS):
        raise ValueError('Invalid measured execution cost')
    if cost['train_us'] <= 0 or cost['audit_us'] <= 0:
        raise ValueError('Execution rates must be positive')


def smooth(previous, observation):
    validate_cost(previous); validate_cost(observation)
    return {k: (4*previous[k] + min(2*previous[k], max((previous[k]+1)//2, observation[k])) + 2)//5
            if previous[k] else observation[k] for k in COST_FIELDS}


def apportion(total, weights):
    """Largest remainder, stable ties by membership order; exact integer sum."""
    if total < 0 or not weights or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Invalid allocation weights')
    shares = [Fraction(total)*w/sum(weights) for w in weights]
    counts = [int(x) for x in shares]
    for i in sorted(range(len(weights)), key=lambda i: (-(shares[i]-counts[i]), i))[:total-sum(counts)]:
        counts[i] += 1
    return counts


def rounded(total, weights, batch):
    units, remainder = divmod(total, batch)
    minimum = 1 if units >= len(weights) else 0
    alloc = apportion(units,weights)
    if minimum:
        for i in range(len(alloc)):
            if alloc[i]:continue
            donor=max(range(len(alloc)),key=lambda j:(alloc[j],-j))
            alloc[donor]-=1;alloc[i]=1
    counts = [n*batch for n in alloc]
    if remainder:
        # Exactly one final short batch; deterministic and no sample loss.
        counts[max(range(len(weights)), key=lambda i: (weights[i], -i))] += remainder
    return counts


def phase_cost(counts, costs, audit):
    train = [c['setup_us'] + n*c['train_us'] if n else 0 for n,c in zip(counts,costs)]
    tail = []
    for i,c in enumerate(costs):
        others = [n for j,n in enumerate(counts) if j != i and n]
        replay = sum(others)*c['audit_us'] + len(others)*c['audit_setup_us'] if audit == 'full' else 0
        tail.append(replay+c['evaluation_us'])
    return max(train)+max(tail)


def allocate(total, batch, costs, policy='compute-aware', audit='full', previous=None):
    if type(total) is not int or not 1 <= total <= 50000 or type(batch) is not int or not 1 <= batch <= 512 or not 1 <= len(costs) <= 8:
        raise ValueError('Invalid allocation budget')
    if policy not in ('equal','compute-aware') or audit not in ('none','full'):
        raise ValueError('Unknown allocation/audit policy')
    for c in costs: validate_cost(c)
    equal = rounded(total, [1]*len(costs), batch)
    baseline = rounded(total, previous, batch) if previous is not None else equal
    # Rescaling is deterministic for a shorter final window; movement is bounded
    # relative to this rescaled, batch-rounded baseline.
    bound = max(batch, total//10)
    minimum = batch if total >= len(costs)*batch else 0
    def feasible(xs):
        return all(x >= minimum and (previous is None or abs(x-b) <= bound) for x,b in zip(xs,baseline))
    objective = lambda xs: phase_cost(xs,costs,audit)
    best = baseline
    if policy == 'compute-aware':
        seeds = [baseline, equal, rounded(total, [Fraction(1,c['train_us']) for c in costs], batch)]
        seen = set(); evaluated = 0
        for seed in seeds:
            if not feasible(seed): continue
            current = seed
            while evaluated < SETTINGS['candidate_limit']:
                candidates = []
                for i in range(len(costs)):
                    for j in range(len(costs)):
                        if i == j or current[i]-batch < minimum: continue
                        xs = current.copy(); xs[i] -= batch; xs[j] += batch
                        key = tuple(xs)
                        if key in seen or not feasible(xs): continue
                        seen.add(key); evaluated += 1
                        candidates.append(xs)
                        if evaluated >= SETTINGS['candidate_limit']: break
                    if evaluated >= SETTINGS['candidate_limit']: break
                better = min(candidates, key=lambda xs:(objective(xs),tuple(xs)), default=current)
                if objective(better) >= objective(current): break
                current = better
            if objective(current) < objective(best): best = current
        if objective(best)*100 > objective(baseline)*98: best = baseline
    else:
        best = equal
    return dict(counts=best,predicted_phase_us=objective(best),equal_phase_us=objective(equal),
                audit_limited=audit=='full' and objective(best)*100 > objective(equal)*98)


def make_plan(config, epoch, round_id, offset, model_digest, costs, evidence, previous=None):
    total = min(config['sync_samples'],config['train_limit']-offset)
    decision = allocate(total,config['batch_size'],costs,config['allocation'],config['audit'],previous)
    return dict(version=1,run_id=config['run_id'],config_digest=digest(config),epoch=epoch,round=round_id,
                offset=offset,total=total,member_ids=[m['id'] for m in config['members']],
                model_digest=model_digest,evidence_digest=digest(evidence),costs=costs,**decision)


def validate_plan(plan, payload, model_digest):
    if not isinstance(plan,dict) or payload.get('plan_digest') != digest(plan):
        raise ValueError('Assignment plan digest mismatch')
    counts=plan.get('counts',[]); ids=plan.get('member_ids',[])
    if plan.get('version') != 1 or len(counts)!=payload['peers'] or len(ids)!=len(counts) or len(set(ids))!=len(ids) or ids!=sorted(ids):
        raise ValueError('Invalid plan membership')
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts)!=plan.get('total'):
        raise ValueError('Invalid plan sample counts')
    if any(plan.get(k)!=payload.get(k) for k in ('run_id','config_digest','epoch','round')) or plan.get('model_digest')!=model_digest:
        raise ValueError('Stale or incorrectly bound assignment')
    if type(plan.get('offset')) is not int or plan['offset']<0 or plan['offset']+sum(counts)>payload['train_limit']:
        raise ValueError('Assignment exceeds epoch')
    peer=payload['peer']
    if not 0<=peer<len(counts) or payload['count']!=counts[peer] or payload['start']!=plan['offset']+sum(counts[:peer]):
        raise ValueError('Task does not match agreed range')
