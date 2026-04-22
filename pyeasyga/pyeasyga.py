# -*- coding: utf-8 -*-
"""
    pyeasyga module

 jpf-x fork: add binary representation of genes in Gene class

"""

import random
import copy
from concurrent import futures
from operator import attrgetter

from six.moves import range

from .constraints import Constraints

class Default:
    GENERATIONS=100
    POPULATION_SIZE=50
    GENE_TYPE=0
    MUTATION_PROBABILITY=.04
    GENE_MUTATION_PROBABILITY=.04
    CROSSOVER_PROBABILITY=0.8
    WORKERS=1
    ELITISM=True
    TOURNAMENT_SPLIT=10
    SELECTION='tournament'

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
    >>> geneX=list(range(0,100))
    >>> geneY=list(range(0,100))
    >>> input_data=[geneX,geneY]
    >>> # using gene_mutation_probability.
    >>> easyga=GeneticAlgorithm(input_data,gene_mutation_probability=0.1)
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
                 population_size=Default.POPULATION_SIZE,
                 generations=Default.GENERATIONS,
                 crossover_probability=Default.CROSSOVER_PROBABILITY,
                 mutation_probability=Default.MUTATION_PROBABILITY,
                 elitism=Default.ELITISM,
                 maximise_fitness=True,
                 verbose=False,
                 random_state=None,
                 gene_mutation_probability=Default.GENE_MUTATION_PROBABILITY,
                 tournament_split=Default.TOURNAMENT_SPLIT,
                 selection=Default.SELECTION,
                 gene_type=Default.GENE_TYPE,
                 constraints=None):
        """Instantiate the Genetic Algorithm.

        :param seed_data: list of lists of possible values for each Gene
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of individual mutation operation
        :param int: random seed. defaults to None
        :param float gene_mutation_probability: probability of bit mutation per generation. ignore mutation_probability.
        :param int tournament_split: split of population for tournament selection, e.g. 4 samples from 25% of population

        """

        self.seed_data = seed_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability if gene_type else 1.
        self.gene_mutation_probability=gene_mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness
        self.verbose = verbose
        self.tournament_split=tournament_split
        self.selection=selection

        # seed random number generator
        self.random = random.Random(random_state)

        self.current_generation = []
        self.fitness_function = None

        self.tournament_size = self.population_size // self.tournament_split
        self.gene_type=gene_type
        if self.selection=='tournament':
            self.selection_function = self.tournament_selection
        elif self.selection=='natural':
            self.selection_function = self.natural_selection
        else:
            self.selection_function=self.random_selection
        self.constraints=Constraints(constraints) if constraints else Constraints([lambda *v: 1])

    @classmethod
    def from_dict(cls,D):
        kwargs={}
        for attr in ['seed_data','population_size','generations','crossover_probability','mutation_probability',\
                        'gene_mutation_probability','elitism','maximise_fitness','verbose','tournament_split',\
                        'tournament_split','selection','tournament_size','gene_type']:
            kwargs.update({attr:D[attr]})
        obj=cls(**kwargs) # note that constraints and fitness_function must be reloaded
        for chromosome in D['chromosomes']:
            obj.current_generation.append(Chromosome.from_dict(chromosome))
        return obj

    def to_dict(self):
        D={'class':self.__class__.__name__}
        for attr in ['seed_data','population_size','generations','crossover_probability','mutation_probability',\
                        'gene_mutation_probability','elitism','maximise_fitness','verbose','tournament_split',\
                        'tournament_split','selection','tournament_size','gene_type']:
            D.update({attr:getattr(self,attr)})
        D['chromosomes']=[]
        for chromosome in self.current_generation:
            D['chromosomes'].append(chromosome.to_dict())

        # note that constraints and fitness_function are missing
        return D

    def load_constraints(self,constraints):
        self.constraints=Constraints(constraints) if constraints else Constraints([lambda *v: 1])
        for chromosome in self.current_generation:
            chromosome.load_constraints(self.constraints)

    def create_individual(self,with_values=None):
        """Create a candidate solution representation.

        e.g. for a bit array representation:

        >>> return [random.randint(0, 1) for _ in range(len(data))]

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :returns: candidate solution representation as a list

        """
        chromosome=Chromosome(constraints=self.constraints)
        for s in self.seed_data:
            gene=Gene(possible_values=s,\
                 gene_type=self.gene_type,\
                 mutation_probability=self.gene_mutation_probability)
            chromosome.append(gene)
        chromosome.initialize()
        return chromosome

    def random_selection(self):
        """Select and return a random member of the population."""
        return self.random.choice(self.current_generation)

    def tournament_selection(self):
        """Select a random number of individuals from the population and
        return the fittest member of them all.
        """
        if self.tournament_size == 0:
            self.tournament_size = 2
        members = self.random.sample(self.current_generation, self.tournament_size)
        members.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)
        return members[0].copy()

    def natural_selection(self):
        """Select an individual from population with probability proportional to fitness
        """
        from numpy import exp,log

        #print([valid.fitness for valid in valids])
        #valids=valids[:-self.tournament_size]
        #members = self.random.sample(population, self.tournament_size)
        valids=[member for member in self.current_generation if abs(member.fitness)!=infinity]
        valids.sort(key=lambda x: x.fitness,reverse=self.maximise_fitness) # best to worst individuals
        extreme_function=min if self.maximise_fitness else max
        extreme=extreme_function(v.fitness for v in valids)
        m=-1**(not self.maximise_fitness)

        def rescale(positives):
            rescaled=[]
            minimum=0.
            maximum=10.
            mx=max(positives)
            mn=min(positives)
            for p in positives:
                v=minimum+(p-mn)*(maximum-minimum)/(mx-mn)
                rescaled.append(v)
            
            return rescaled
        def cdf(valids):

            FUN=exp
            positive=[m*(member.fitness-extreme) for member in valids]
            positive=rescale(positive)
            pdfbar=[]
            s=0.
            for im,member in enumerate(valids):
                v=FUN(positive[im])
                pdfbar.append(v)
                s+=v
            pdfbar=[c/s for c in pdfbar]
            cdfbar=[]
            s=0.
            for v in pdfbar:
                cdfbar.append(v+s)
                s+=v
            cdfbar=[v/s for v in cdfbar]
            return cdfbar

        cdfbar=cdf(valids)
        #print(list(zip([valid.fitness for valid in valids],cdfbar)))
        rn=self.random.random()
        selected=None
        #select
        for ix,v in enumerate(valids):
            if rn<cdfbar[ix]:
                selected=v
                break
        #print('selected: ',selected,extreme)
        return selected

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual()
            individual = Chromosome(genes,constraints=self.constraints)
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
                     values=individual.phenotype
                     individual.fitness = self.fitness_function(values)
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
            individuals=[]
            for individual in self.current_generation:
                try:
                    individuals.append(individual.phenotype)
                except InvalidGene:
                    individual.fitness=-infinity if self.maximise_fitness else infinity
                    individuals.append(individual.values)

            with executor as pool:
                results = pool.map(self.fitness_function, individuals)

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
        elite = self.current_generation[0].copy()
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = self.selection_function().copy()
            parent_2 = self.selection_function().copy()

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = self.random.random() < self.crossover_probability

            if can_crossover:
                child_1, child_2 = parent_1.crossover(parent_2)

            if self.random.random() < self.mutation_probability:
                child_1.mutate()
            if self.random.random() < self.mutation_probability:
                child_2.mutate()

            if child_1.is_valid:
                new_population.append(child_1)

            if len(new_population) < self.population_size and child_2.is_valid:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite
        self.current_generation = new_population

    def create_first_generation(self, n_workers=Default.WORKERS, parallel_type="processing"):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness(
            n_workers=n_workers, parallel_type=parallel_type
        )
        self.rank_population()

    def create_next_generation(self, n_workers=Default.WORKERS, parallel_type="processing"):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        self.calculate_population_fitness(
            n_workers=n_workers, parallel_type=parallel_type
        )
        self.rank_population()

    def run(self, n_workers=1, parallel_type="processing"):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation(
                n_workers=n_workers, parallel_type=parallel_type
            )

        try:
            for _ in range(1, self.generations):
                self.create_next_generation(
                    n_workers=n_workers, parallel_type=parallel_type
                )
                print(f'generation {_:04d} best[{self.best_individual()}] number uniques[{self.number_unique}]')
        except KeyboardInterrupt:
            print(f'generation {_:04d} best[{self.best_individual()}] number uniques[{self.number_unique}]')
            print('unique members: ')
            print(self.uniques, 'number ',len(self.uniques))

    @property
    def uniques(self):
        """Return list of unique individuals.
        """
        uqs=[]
        for x in self.current_generation:
            if any([x==u for u in uqs]): continue
            uqs.append(x)
        return uqs
#        return list(set(tuple(x.phenotype) for x in self.current_generation))

    @property
    def number_unique(self):
        return len(self.uniques)   

    def best_individual(self,as_phenotype=False):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return best

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        for member in self.current_generation:
            try:
                phenotype=member.phenotype
            except InvalidGene:
                phenotype=None
            yield (member.fitness, phenotype)

class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes=None, random_state=None, constraints=None,fitness=0):
        """Initialise the Chromosome."""
        self.genes=[]
        if genes:
            for gene in genes:
                if isinstance(gene,Gene):
                    self.genes.append(gene)
                else: # must be possible values for gene desired
                    new_gene=Gene(gene)
                    new_gene.initialize_value()
                    self.genes.append(new_gene)

        self.fitness = fitness
        # seed random number generator
        self.random = random.Random(random_state)
        self.constraints=Constraints(constraints) if constraints else []

    def __getitem__(self,key):
        return self.genes[key]

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return f'fitness={self.fitness} [{",".join([repr(gene) for gene in self.genes])}]'

    def __len__(self):
        return len(self.genes)

    def __eq__(self,other):
        return all([s.value == other.genes[i].value for i,s in enumerate(self.genes)])

    def initialize(self,to_value=None):
        if to_value is None:
            for i in range(len(self.genes)):
                self.genes[i].initialize_value()
        else:
            for i in range(len(self.genes)):
                self.genes[i].initialize_value(to_value=to_value[i])

    def load_constraints(self,constraints):
        self.constraints=Constraints(constraints) if constraints else []

    @property
    def phenotype(self):
        return [gene.value for gene in self.genes]

    def crossover(self,other):
        """Crossover (mate) two parents to produce two children.

        :param parent_1: candidate solution representation (list)
        :param parent_2: candidate solution representation (list)
        :returns: tuple containing two children

        """
        parent_1=self
        parent_2=other
        child_1=Chromosome(constraints=self.constraints)
        child_2=Chromosome(constraints=self.constraints)
        for index in range(len(parent_1)):
            if self.random.random()<.5:
              child_1.append(parent_1[index].copy())
              child_2.append(parent_2[index].copy())
            else:
              child_1.append(parent_2[index].copy())
              child_2.append(parent_1[index].copy())
        return child_1, child_2

    def mutate(self):
        for gene in self.genes:
            gene.mutate()

    def append(self,gene):
        self.genes.append(gene)

    @property
    def is_valid(self):
        return self.constraints(*self.phenotype)

    def copy(self):
        new=Chromosome()
        new.genes=[gene.copy() for gene in self.genes]
        new.fitness = self.fitness
        new.random = random.Random(None)
        new.constraints=self.constraints
        return new

    @classmethod
    def from_dict(cls,D):
        kwargs={}
        for attr in ['fitness']:
            kwargs.update({attr:D[attr]})
        obj=cls(**kwargs) # note that constraints must be reloaded
        for gene in D['genes']:
            obj.genes.append(Gene.from_dict(gene))
        return obj

    def to_dict(self):
        D={'class':self.__class__.__name__}
        for attr in ['fitness',]:
            D.update({attr:getattr(self,attr)})
        D['genes']=[]
        for gene in self.genes:
            D['genes'].append(gene.to_dict())
        # note that constraints are missing
        return D

class Gene(object):
    """
Binary representation of a set of values.
"""
    def __init__(self,possible_values=None,\
                 gene_type=Default.GENE_TYPE,\
                 mutation_probability=Default.GENE_MUTATION_PROBABILITY,\
                 random_state=None):
        from math import floor,log
        self._possible_values=possible_values if possible_values is not None else [] # duplicate values magnify probability of occurrence
        self._len=len(possible_values)   # this is the number that requires binary representation
        self._binary_digits=floor(log(self._len,2))+1 # binary digits
        self.mutation_probability=mutation_probability
        self._gene_type=gene_type

        # current value attributes
        self._index=None # index to current value from list of possible values
        self._value=None # current value at _index
        self._bin_value=None # current binary value at index _index
        self.mutate = self._mutate_gene if gene_type else self._mutate_bits

        self.random = random.Random(random_state)

    def __getitem__(self,key):
        return self._possible_values[key]

    def __repr__(self):
        return f'Gene(_value={str(self._value)})'

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self,val):
        self.initialize_value(to_value=val)

    @property
    def binary_digits(self):
        return self._binary_digits

    def get(self,n):
        """Get binary gene value at index n"""
        assert n<self._len+1,\
               f'gene value not defined at index {n}'
        a=[int(i) for i in list(bin(n))[2:]]
        a=[0 for i in range(self._binary_digits-len(a))]+a
        return a

    def get_binary(self):
        return self.get(self._index)

    def initialize_value(self,to_value=None):
        import random
        if to_value is None:
            self._index=random.randint(0,self._len-1)
            self._value=self._possible_values[self._index]
        else:
            self._index=self._possible_values.index(to_value)
            self._value=to_value
        self._bin_value=self.get(self._index)

    def _mutate_gene(self):
        """Reverse the bit of a random index in an individual."""
        if self.random.random()<self.mutation_probability:
            mutate_index = self.random.randrange(self._len)
            self._value=self._possible_values[mutate_index]

    def _mutate_bits(self):
        """Reverse the bits of genes with gene_mutation_probability."""
        # print(f'mutating gene {self}')
        while True:
            try:
                # print(f'before mutate : {self._bin_value}')
                for mutate_index in range(len(self._bin_value)):
                    if self.random.random()<self.mutation_probability:
                        # print(f'mutate_index={mutate_index} of {len(self._bin_value)}')
                        self._bin_value[mutate_index] = (0, 1)[self._bin_value[mutate_index] == 0]
                # print(f' after mutate : {self._bin_value}')
                self._value=value_from_bits(self._bin_value,self._possible_values)
                break
            except InvalidGene:
                continue

    @property
    def is_valid(self):
        try:
            self.phenotype
        except InvalidGene:
            return False
        return True

    @property
    def phenotype(self):
        if self._gene_type:
            return self.value
        else:
            self._value=value_from_bits(self._bin_value,self._possible_values)
            return self._value

    def copy(self):
        new=Gene(possible_values=self._possible_values,mutation_probability=self.mutation_probability,\
                    gene_type=self._gene_type,)
        for attr in ['_index','_value','_bin_value']:
            setattr(new,attr,getattr(self,attr))
        new.random = random.Random(None)
        return new

    @classmethod
    def from_dict(cls,D):
        kwargs={}
        for attr in ['_possible_values','_index','_value','_bin_value','mutation_probability','_gene_type']:
            kwargs.update({attr:D[attr]})
        obj=cls(**kwargs) # note that constraints must be reloaded
        return obj

    def to_dict(self):
        D={'class':self.__class__.__name__}
        for attr in ['_possible_values','_index','_value','_bin_value','mutation_probability','_gene_type']:
            D.update({attr:getattr(self,attr)})
        return D

class InvalidGene(Exception):
    pass

def value_index_from_bits(bits):
    """Translate bits to phenotype."""
    s=''.join([str(b) for b in bits])
    return int(s,base=2)

def value_from_bits(bits,values):
    """Translate bits to phenotype."""
    s=''.join([str(b) for b in bits])
    key=int(s,base=2)
    try:
        value=values[key]
    except IndexError:
        raise InvalidGene()

    return value

def bits_from_value_index(value_index):
    return [int(b) for b in bin(value_index)[2:]]

def phenotype_from_genes(member_genes,seed_data):
    """Translate genes to phenotype."""
    i=0
    values=[]
    for _ in seed_data:
        if isinstance(_,Gene):
            bin_gene=member_genes[i:i+_._binary_digits]
            s=''.join([str(x) for x in bin_gene])
            key=int(s,base=2)
            try:
                value=_[key]
            except IndexError:
                raise InvalidGene()
            values.append(value)
            i+=_._binary_digits
        else:
            values.append(member_genes[i])
            i+=1
    return values
