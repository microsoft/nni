# TODO: needs to be adapted to new API


class RegularizedEvolution:
    def __init__(self, search_space,
                 concurrency, population_size, sample_size, cycles, mutation_prob,
                 reward_fn, command, setup):
        self.search_space = search_space
        self.concurrency = concurrency
        self.population_size = population_size
        self.command = command
        self.setup = setup
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob
        self.reward_fn = reward_fn
        assert self.cycles >= self.population_size >= self.sample_size

        self.population = collections.deque()

    def train_and_eval(self, config):
        pid = get_trial_manager().submit_new_trial(self.command, config, self.setup)

        while True:
            try:
                metrics = get_trial_manager().query_metrics(pid)
                if metrics is not None:
                    break
                time.sleep(5)
                continue
            except TrialFailed:
                _logger.warning(f'Config: {config}. Trial failed and use -inf as metrics.')
                metrics = float('-inf')
                break
        return self.reward_fn(config, metrics)

    def random_config(self):
        config = {}
        for k, v in SearchSpaceUtils.flatten_search_space(self.search_space).items():
            config[k] = v.random()
        _logger.info(f'Generated random config: {config}')
        return SearchSpaceUtils.restore_config(config, self.search_space)

    def mutate_config(self, parent_config):
        parent_config = SearchSpaceUtils.flatten_config(parent_config)
        config = {}
        for k, v in SearchSpaceUtils.flatten_search_space(self.search_space).items():
            config[k] = parent_config[k]
            if random.uniform(0, 1) < self.mutation_prob:
                config[k] = v.random(excludes=[parent_config[k]])
        _logger.info(f'Generated mutated config: {config}')
        return SearchSpaceUtils.restore_config(config, self.search_space)

    def import_(self, individuals):
        self.individuals = sorted(individuals, key=lambda i: i.reward)[-self.population_size:]
        random.shuffle(self.individuals)
        _logger.info(f'Imported individuals: {self.individuals}')

    def _run_random(self):
        individual = Individual(self.random_config(), None)
        individual.reward = self.train_and_eval(individual.config)
        self.population.append(individual)

    def _run_mutation(self):
        # Sample randomly chosen models from the current population.
        try:
            _lock.acquire()
            samples = copy.deepcopy(self.population)
        finally:
            _lock.release()
        random.shuffle(samples)
        samples = list(samples)[:self.population_size]
        parent = max(samples, key=lambda i: i.reward)

        individual = Individual(self.mutate_config(parent.config), None)
        individual.reward = self.train_and_eval(individual.config)
        try:
            _lock.acquire()
            self.population.append(individual)
            self.population.popleft()
        finally:
            _lock.release()

    def _wait_for_futures_and_shutdown(self, futures, pool):
        for i in futures:
            try:
                i.result()
            except:
                traceback.print_exc()
                for k in futures:
                    k.cancel()
                pool.shutdown(wait=True)
                raise
        pool.shutdown()

    def run(self):
        # Initialize the population with random models.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        fs = [pool.submit(self._run_random) for _ in range(self.population_size - len(self.population))]
        self._wait_for_futures_and_shutdown(fs, pool)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        fs = [pool.submit(self._run_mutation) for _ in range(self.cycles - self.population_size)]
        self._wait_for_futures_and_shutdown(fs, pool)
