import pickle
from pprint import pprint

with open(f'../log/fitness_cache.pkl', 'rb') as f:
    fitness_results = pickle.load(f)

pprint(fitness_results)
