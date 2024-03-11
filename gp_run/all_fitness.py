import pickle
from pprint import pprint

import pandas as pd

with open(f'../log/fitness_cache.pkl', 'rb') as f:
    fitness_results = pickle.load(f)

pprint(fitness_results)

# 转成DataFrame
df = pd.DataFrame.from_dict(fitness_results, orient='index').reset_index()
print(df)

df.to_excel(f'../log/fitness_cache.xlsx')
