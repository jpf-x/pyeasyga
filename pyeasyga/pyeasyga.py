# -*- coding: utf-8 -*-
"""
    pyeasyga module

"""

import random
import copy
from concurrent import futures
from operator import attrgetter

from six.moves import range

infinity=float('inf')

class GeneticAlgorithm(object):
    """Genetic Algorithm class.

    This is the main class that controls the functionality of the Genetic
    Algorithm.

    A simple example of usage:

    >>> # Select only two items from the list and maximise profit
    >>> from pyeasyga.pyeasyga import GeneticAlgorithm
    >>> input_data = [('pear', 50), ('apple', 35), ('banana', 40)]
    >>> easyga = GeneticAlgorithm(input_data)
    >>> def fitness (member, data):
    >>>     return sum([profit for (selected, (fruit, profit)) in
    >>>                 zip(member, data) if selected and
    >>>                 member.count(1) == 2])
    >>> easyga.fitness_function = fitness
    >>> easyga.run()
    >>> print(easyga.best_individual())


    A simple example using the Gene class:
    >>> # Find maximum of function (fitness)
    >>> from pyeasyga.pyeasyga import GeneticAlgorithm,Gene
    >>> # Define all possible values of each variable
    >>> geneX=Gene(list(range(0,100)))
    >>> geneY=Gene(list(range(0,100)))
    >>> input_data=[geneX,geneY]
    >>> # using bit_mutation_probability.
    >>> easyga=GeneticAlgorithm(input_data,bit_mutation_probability=0.1)
    >>> def fitness(phenotype,data):
    >>>     x,y=phenotype
    >>>     return -x**2-y**2+2*x
    >>> easyga.fitness_function=fitness
    >>> easyga.run()
    >>> print(easyga.best_individual())
    >>> # yields solution: (1, [1,0]), the fitness function maximum is 1 when x=1,y=0.

    """

    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 elitism=False,
                 maximise_fitness=True,
                 verbose=False,
                 random_state=None,
                 bit_mutation_probability=None,
                 tournament_split=10,):
        """Instantiate the Genetic Algorithm.

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of individual mutation operation
        :param int: random seed. defaults to None
        :param float bit_mutation_probability: probability of bit mutation per generation. ignore mutation_probability.
        :param int tournament_split: split of population for tournament selection, e.g. 4 samples from 25% of population

        """

        self.seed_data = seed_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        if not bit_mutation_probability:
            self.mutation_probability = mutation_probability
        else:
            self.mutation_probability = 1.0
        self.bit_mutation_probability=bit_mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness
        self.verbose = verbose
        self.tournament_split=tournament_split

        # seed random number generator
        self.random = random.Random(random_state)

        self.current_generation = []

        def create_individual(seed_data):
            """Create a candidate solution representation.

            e.g. for a bit array representation:

            >>> return [random.randint(0, 1) for _ in range(len(data))]

            :param seed_data: input data to the Genetic Algorithm
            :type seed_data: list of objects
            :returns: candidate solution representation as a list

            """
            ret=[]
            for i in range(len(seed_data)):
                if isinstance(seed_data[i],Gene):
                    seed_data[i].initialize_value()
                    ret+=seed_data[i].get_binary()
                else:
                    ret+=[self.random.randint(0,1)]
            return ret

        def crossover(parent_1, parent_2):
            """Crossover (mate) two parents to produce two children.

            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children

            """
            child_1=[]
            child_2=[]
            for index in range(len(parent_1)):
                if self.random.random()<.5:
                  child_1.append(parent_1[index])
                  child_2.append(parent_2[index])
                else:
                  child_1.append(parent_2[index])
                  child_2.append(parent_1[index])
            return child_1, child_2

        def _mutate_genes(genes):
            """Reverse the bit of a random index in an individual."""
            mutate_index = self.random.randrange(len(genes))
            genes[mutate_index] = (0, 1)[genes[mutate_index] == 0]

        def _mutate_bits(bits):
            """Reverse the bits of genes with bit_mutation_probability."""
            for mutate_index in range(len(bits)):
                if self.random.random()<self.bit_mutation_probability:
                    bits[mutate_index] = (0, 1)[bits[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return self.random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = self.random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.fitness_function = None
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // self.tournament_split
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        if not self.bit_mutation_probability:
            self.mutate_function = _mutate_genes
        else:
            self.mutate_function = _mutate_bits
        self.selection_function = self.tournament_selection

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual(self.seed_data)
            individual = Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self, n_workers=1, parallel_type="processing"):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        # If using a single worker, run on a simple for loop to avoid losing
        # time creating processes.
        if n_workers == 1:
            for individual in self.current_generation:
                 try:
                     phenotype=individual.as_phenotype(self.seed_data)
                     individual.fitness = self.fitness_function(
                         phenotype, self.seed_data)
                 except InvalidGene:
                     phenotype=None
                     individual.fitness=-infinity if self.maximise_fitness else infinity
        else:

            if "process" in parallel_type.lower():
                executor = futures.ProcessPoolExecutor(max_workers=n_workers)
            else:
                executor = futures.ThreadPoolExecutor(max_workers=n_workers)

            # Create two lists from the same size to be passed as args to the
            # map function.
            genes=[]
            data=[]
            individuals=[]
            for individual in self.current_generation:
                try:
                    phenotype=individual.as_phenotype(self.seed_data)
                    genes.append(phenotype)
                    data.append(self.seed_data)
                    individuals.append(individual)
                except InvalidGene:
                    individual.fitness=-infinity if self.maximise_fitness else infinity

            with executor as pool:
                results = pool.map(self.fitness_function, genes, data)

            for individual, result in zip(individuals, results):
                individual.fitness = result

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = self.random.random() < self.crossover_probability
            can_mutate = self.random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    def create_first_generation(self, n_workers=1, parallel_type="processing"):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness(
            n_workers=n_workers, parallel_type=parallel_type
        )
        self.rank_population()

    def create_next_generation(self, n_workers=1, parallel_type="processing"):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        self.calculate_population_fitness(
            n_workers=n_workers, parallel_type=parallel_type
        )
        self.rank_population()
        if self.verbose:
            print("Fitness: %f" % self.best_individual()[0])

    def run(self, n_workers=1, parallel_type="processing"):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation(
                n_workers=n_workers, parallel_type=parallel_type
            )

        for _ in range(1, self.generations):
            self.create_next_generation(
                n_workers=n_workers, parallel_type=parallel_type
            )

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return (best.fitness, best.as_phenotype(self.seed_data))

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        report=[]
        for member in self.current_generation:
            try:
                phenotype=member.as_phenotype(self.seed_data)
            except InvalidGene:
                phenotype=None
            report.append((member.fitness, phenotype))

        return tuple(report)


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes=[]
        for gene in genes:
            if isinstance(gene,Gene):
                self.genes+=gene.get_binary()
            else:
                self.genes.append(gene)
        self.fitness = 0

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))

    def as_phenotype(self,seed_data):
        phenotype=phenotype_from_genes(self.genes,seed_data)
        return phenotype
        

class Gene(object):
    """
Binary representation of a set of values.
"""
    def __init__(self,possible_values):
        from math import floor,log
        self._possible_values=list(set(possible_values))
        self._len=len(possible_values)   # this is the number that requires binary representation
        self._digits=floor(log(self._len,2))+1 # binary digits
        self._index=None # index to current value from list of possible values
        self._value=None # current value at _index

    def __getitem__(self,key):
        return self._possible_values[key]

    @property
    def value(self):
        return self._value

    def get(self,n):
        """Get binary gene value at index n"""
        assert n<self._len+1,\
               f'gene value not defined at index {n}'
        a=[int(i) for i in list(bin(n))[2:]]
        a=[0 for i in range(self._digits-len(a))]+a
        return a

    def get_binary(self):
        return self.get(self._index)

    def initialize_value(self):
        import random
        self._index=random.randint(0,self._len-1)
        self._value=self._possible_values[self._index]

class InvalidGene(Exception):
    pass

def phenotype_from_genes(member_genes,seed_data):
    """Translate genes to phenotype."""
    i=0
    values=[]
    for _ in seed_data:
        if isinstance(_,Gene):
            bin_gene=member_genes[i:i+_._digits]
            s=''.join([str(x) for x in bin_gene])
            key=int(s,base=2)
            try:
                value=_[key]
            except IndexError:
                raise InvalidGene()
            values.append(value)
            i+=_._digits
        else:
            i+=1
            values.append(_)
    return values
