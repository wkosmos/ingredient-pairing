import numpy as np
import pandas as pd


with np.load('data/simplified-recipes-1M.npz', allow_pickle=True) as data:
    recipes = data['recipes']
    ingredients = data['ingredients']

word_recipes_10k = [ingredients[recipes[i]] for i in range(10000)]

recipes_df = pd.DataFrame(word_recipes_10k)

recipes_df.head(10)