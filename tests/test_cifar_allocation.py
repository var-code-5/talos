import unittest
from mtcp.cifar.allocation import allocate, apportion, phase_cost, smooth


def cost(us):
    return dict(train_us=us,setup_us=0,audit_us=us,audit_setup_us=0,evaluation_us=0)


class AllocationTests(unittest.TestCase):
    def test_proportional_and_synthetic_wait(self):
        costs=[cost(1),cost(2)]
        result=allocate(9600,32,costs,audit='none')
        self.assertEqual(result['counts'],[6400,3200])
        equal_wait=abs(4800-9600)
        work=[n*c['train_us'] for n,c in zip(result['counts'],costs)]
        self.assertLessEqual(max(work)-min(work),equal_wait/2)
        self.assertLess(phase_cost(result['counts'],costs,'none'),phase_cost([4800,4800],costs,'none'))

    def test_full_audit_and_equal(self):
        self.assertEqual(allocate(9600,32,[cost(1),cost(2)])['counts'],[4800,4800])
        self.assertTrue(allocate(9600,32,[cost(1),cost(2)])['audit_limited'])
        self.assertEqual(allocate(9600,32,[cost(1),cost(2)],policy='equal')['counts'],[4800,4800])

    def test_coverage_rounding_extreme_and_bounds(self):
        for peers in range(1,9):
            for total in (1,7,31,32,33,101,10000):
                r=allocate(total,32,[cost(1 if i==0 else 10000) for i in range(peers)],audit='none')
                self.assertEqual(sum(r['counts']),total)
                self.assertTrue(all(n>=0 for n in r['counts']))
        r=allocate(10000,32,[cost(1),cost(1000)],audit='none',previous=[5000,5000])
        self.assertLessEqual(abs(r['counts'][0]-5008),1000)

    def test_global_indices(self):
        from mtcp.cifar.training import global_indices
        for peers in range(1,9):
            for size in (7,33,101):
                seen=[]
                for start in range(0,size,32):
                    counts=allocate(min(32,size-start),8,[cost(i+1) for i in range(peers)],audit='none')['counts']
                    offset=start
                    for n in counts:
                        seen.extend(global_indices(42,size,1,offset,n).tolist());offset+=n
                self.assertEqual(len(seen),size)
                self.assertEqual(len(set(seen)),size)
                self.assertEqual(set(seen),set(global_indices(42,size,0,0,size)))

    def test_smoothing_and_invalid(self):
        self.assertEqual(smooth(cost(100),cost(10000))['train_us'],120)
        self.assertEqual(smooth(cost(100),cost(1))['train_us'],90)
        self.assertEqual(apportion(7,[1,1,1]),[3,2,2])
        with self.assertRaises(ValueError): allocate(10,32,[cost(0)])
        with self.assertRaises(ValueError): allocate(10,32,[cost(float('nan'))])


if __name__=='__main__':unittest.main()
