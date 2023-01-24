import numpy as np

class Roulette:
    class Mutator:
        def __init__(self, name, difference_score_total=0,total_select_times= 0):
            self.name = name
            self.difference_score_total = difference_score_total
            self.total_select_times = total_select_times

    def __init__(self,mutate_ops=None):
        from src.DrFuzz_mutop import get_mutation_ops_name
        self.mutate_ops = get_mutation_ops_name()

    def choose_mutator(self,mu1=None):
        return np.random.choice(self.mutate_ops)


class MCMC:
    class Mutator:
        def __init__(self, name, difference_score_total=0,total_select_times= 0):
            self.name = name
            self.difference_score_total = difference_score_total
            self.fidelity_case_num = 0
            self.total_select_times = total_select_times

        @property
        def score(self):
            if self.total_select_times == 0:
                return 0
            else:
                rate = self.difference_score_total * self.fidelity_case_num / (self.total_select_times * self.total_select_times)
            return rate

    def __init__(self,mutate_ops=None):
        if mutate_ops is None:
            from src.DrFuzz_mutop import get_mutation_ops_name
            mutate_ops = get_mutation_ops_name()
        self.p = 1 / len(mutate_ops)
        print(mutate_ops)
        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self):
        mus = {}
        for mu in self._mutators:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None:
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        else:
            self.sort_mutators()
            k1 = self.index(mu1)
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self._mutators[k2]
            return mu2.name

    def sort_mutators(self):
        import random
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self._mutators):
            if mu.name == mutator_name:
                return i
        return -1