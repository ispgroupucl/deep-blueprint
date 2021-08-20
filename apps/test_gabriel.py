import functools
import random
import numpy as np


def pseudo_random(seed=0, evolutive=True, input_dependent=True, loop=0):
    """ Decorator Factory to control randomness of decorated function, depending on the arguments the function is called with.
        Arguments:
            evolutive (bool): random seed is incremented each time the function is called with given arguments
            input_dependent (bool): initial random seed is unique given the function's arguments
        contingent: random depend on the key
    """

    def decorator(f):
        class DecoratorFactory:
            def __init__(
                self, f, *, seed=0, evolutive=True, input_dependent=True, loop=0
            ):
                self.f = f
                self.seed = seed
                self.__history = {}
                self.__evolutive = evolutive
                self.__input_dependent = input_dependent
                self.__loop = loop

            def reset_seed(self):
                self.__history = {}

            @functools.wraps(f)
            def __call__(self, *args, **kwargs):
                key = repr([self.f.__name__, args, kwargs])
                self.__history.setdefault(key, 0)
                # backup random state
                random_state = random.getstate()
                np_random_state = np.random.get_state()
                call_seed = self.__history[key] + (
                    0 if not self.__input_dependent else hash(key)
                )
                # print(key, self.seed, self.__history[key], call_seed)
                # set random state
                random.seed(self.seed + call_seed)
                np.random.seed((self.seed + call_seed) & 0xFFFFFFFF)
                result = self.f(*args, **kwargs)
                # restore random state
                random.setstate(random_state)
                np.random.set_state(np_random_state)
                if self.__evolutive:
                    self.__history[key] += 1
                    if self.__loop:
                        self.__history[key] %= self.__loop
                return result

        return DecoratorFactory(
            f,
            seed=seed,
            evolutive=evolutive,
            input_dependent=input_dependent,
            loop=loop,
        )

    return decorator


for consistant in [True, False]:
    for incremental in [True, False]:

        @pseudo_random(evolutive=consistant, input_dependent=incremental)
        def get_random_number(*args):
            return random.randint(0, 9)

        print(
            f"\nincremental: {incremental}\tconsistant: {consistant}\t(10 iterations)\n{'-'*63}"
        )
        for arg in range(10):
            print(f"arguments='{arg}'", end=":  ")
            for i in range(10):
                print(get_random_number(arg), end=" ")
            print("")
