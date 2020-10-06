import numpy as np
import pandas as pd
import itertools
import pickle

with np.load('data/simplified-recipes-1M.npz', allow_pickle=True) as data:
    recipes = data['recipes']
    ingredients = data['ingredients']

word_recipes_10k = [ingredients[recipes[i]] for i in range(10000)]
recipes_10k = [recipes[i] for i in range(10000)]
recipes_1k = [recipes[i] for i in range(1000)]
recipes_100 = [recipes[i] for i in range(100)]


def get_combs(in_lst, comb_len=2):
    """Get possible combinations of length n from input list.
    
    Args:
    in_lst (list): list to get combinations from
    comb_len (int): length of combinations (default 2)

    Returns:
    dict where keys are combinations and values are counts
    
    """
    all_combs = {}

    for recipe in in_lst:
        # get all combinations in recipe
        combs = itertools.combinations(recipe, comb_len)
        for comb in combs:
            if comb not in all_combs:
                all_combs[comb] = 1
            else:
                all_combs[comb] += 1

            

    return all_combs


def recipe_words(recipe):
    """Get the ingredient names for a recipe.
    
    Args:
    recipe (np arr): input recipe as np array of ingredient indices

    Returns:
    input recipe as list of ingredient strings
    
    """
    return [ingredients[i] for i in recipe]


def dict_to_pickle(dict, path):
    """Write a dictionary to a pickle at specified path.
    
    Args:
    dict: dictionary to write to pickle
    path (str): location to save pickle

    Returns:
    nothing
    """
    with open(path, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved ' + str(path))


def NPMI(ing1, ing2, recipes, num_recipes):
    """Calculate the normalized point-wise mutual information of two ingredients."""

    ing1_count = 0
    ing2_count = 0
    both_count = 0

    print('counting ingredients')
    for recipe in recipes:
        if ing1 in recipe:
            ing1_count += 1
        if ing2 in recipe:
            ing2_count += 1
        if ing1 in recipe and ing2 in recipe:
            both_count += 1
        

    p_of_ing1 = ing1_count / num_recipes

    p_of_ing2 = ing2_count / num_recipes

    p_of_ing_1_and_2 = both_count / num_recipes

    # calculate PMI
    pmi = np.log10(p_of_ing_1_and_2 / (p_of_ing1 * p_of_ing2))

    # normalize PMI
    nmpi = pmi / - (np.log10(p_of_ing_1_and_2))

    return nmpi



if __name__ == "__main__":

    # get existing ingredient combinations
    existing_combs_counts = get_combs(recipes)
    
    # get all possible combs of ingredients
    all_combs = itertools.combinations(recipes, 2)

    dict_to_pickle(all_combs, 'data/all_combs.pickle')


    # existing_combs = [comb for comb, val in existing_combs_counts.items()]
    # dict_to_pickle(existing_combs, 'data/existing_combs.pickle'
    
    # dict_to_pickle(existing_combs_counts, 'data/existing_combs_counts.pickle')
    # accumulate NPMI for each existing combination
    # comb_scores = {}

    # num_recipes = len(recipes)

    # for comb in existing_combs:
    #     comb_scores[comb] = NPMI(comb[0], comb[1], recipes, num_recipes)
    #     print(comb)

    # dict_to_pickle(comb_scores, 'data/comb_scores.pickle')