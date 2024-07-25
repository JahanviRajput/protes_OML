from opti import Opti
import sys
sys.path.append("../")
from protes import protes_fed_learning

try:
    import sys
    sys.path.append("../")
    from protes import protes_fed_learning
    with_protes = True
except Exception as e:
    with_protes = False


class OptifedProtes(Opti):
    def __init__(self, name='fedprotes', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, k=100, k_gd=1, lr=5.E-2, r=5, P=None,
             seed=0, nbb = 10):
        self.opts_k = k
        self.opts_k_gd = k_gd
        self.opts_lr = lr
        self.opts_r = r
        self.opts_P = P
        self.opts_seed = seed
        self.opts_nbb = nbb

    def _init(self):
        if not with_protes:
            self.err = 'Need "fed protes" module'
            return

    def _optimize(self):
        protes_fed_learning(self.f_batch, self.d, self.n[0], self.m_max, k=self.opts_k, nbb = self.opts_nbb, P=self.opts_P,
         k_gd=self.opts_k_gd, lr=self.opts_lr, r=self.opts_r, is_max=self.is_max,
            seed=self.opts_seed)
        
print(with_protes)
        
        

