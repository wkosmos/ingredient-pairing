# Food Pairing Prediction
[Project Proposal](#docs/project-proposal.md)

Inspired by [KitcheNette (2019)](https://arxiv.org/pdf/1905.07261.pdf)
![cover image pear hearts](https://github.com/wkosmos/capstone2/blob/master/images/cover.png)

# Contents


# Introduction
This project's goal was to generate new recipes as lists of ingredients, based on a large dataset of existing recipes. Because there is such a ingredients in the world, only a small percentage of possible combinations have been used by chefs and food researchers, and the goal was to generate useful suggestions of new combinations to investigate.

# Dataset and Processing
The dataset chosen was a simplified version of Recipes 1M+, where each recipe had been simplified into a list of ingredients. <br>
This was represented in two numpy arrays: 
- `ingredients` the top 3500 ingredients in the full dataset as strings
- `recipes` arrays of ingredients represented as indices of the ingredients array

Since predicting a full recipe without knowing the number of ingredients it should have would be very complex, and biased toward common combinations of ingredients, I decided to make a pivot toward predicting ingredient pairings. <br>
A sensible way to approach ingredient pairings seemed to be predicting a 'score' of the pairings, which made a relatively approachable supervised regression problem.
_note: The end goal would be semi-supervised version of the problem, predicting the scores of unknown ingredient pairings_ <br>

## Duplicate Ingredient Cleaning
**Example Recipe:** <br> `['bottle', 'bouillon', 'carrots', 'celery', 'chicken bouillon',
        'cilantro', 'clam juice', 'cloves', 'fish', 'garlic',
        'medium shrimp', 'olive', 'olive oil', 'onion', 'pepper',
        'pepper flakes', 'red pepper', 'red pepper flakes', 'salt',
        'sherry', 'shrimp', 'stewed tomatoes', 'tomatoes', 'water',
        'white', 'white wine']` <br>
Many recipes contained a lot of duplicate split and unsplit ingredients which would clearly end up with skewed pairing scores. I worked hard on removing these, but unfortunately could not remove the unwanted ones and not remove too many 'correct' ingredients or remove too much information without manually going through the recipes list and dropping ingredients.


## Scoring
Each pairing present in the dataset needed to be scored in order for the score to be predicted, and KitcheNette gave an excellent metric to use, called Normalized Point-wise Mutual Information. 

PMI(x; y) = log p(x, y) / p(x)p(y)

NMPI(x; y) = pmi(x; y) / -(log p(x, y))

The advantage of this is that it compensates for pairings of popular ingredients, which would otherwise outweigh pairings of uncommon ingredients by having larger overall counts. 

Unfortunately, this calculation (represented below) would have taken around 1800 hours on my laptop due to the algorithmic complexity of the for loop. If I had reached this point in the investigation earlier in the week I could have used AWS, but I decided it would be too much of a tangent and used simple counts of pairings as a score.

```python
def NPMI(ing1, ing2, recipes, num_recipes):
    """Calculate the normalized point-wise mutual information of two ingredients.
  
    """

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
 ```
 ## Inputs
 KitcheNette encoded their ingredient pairings into pairs of 300-dimension word vectors. I read about this technique, which seems to have been pioneered by Wolfram Alpha, and it was not clear how it applied to the project, so I input the pairings as simple integers between 1 and 3500
 

# Model and Experiment

![siamese neural networks](https://github.com/wkosmos/capstone2/blob/master/images/siamese_nn.png)

KitcheNette used Siamese Neural Networks in their model and I did some detailed reading on the architecture before deciding to start with a simpler design. They didn't explain why they chose this architecture, and from my reading I couldn't see how it applied.


Based on some similar projects, I decided to use an MLP to predict the pairings' scores. I used the `Sequential` model from Keras, and the same optimizer and activation functions (adam and ReLU) as KitcheNette, and started with two small 16-neuron dense layers. This performed consistently better the more epochs I ran it over, up to the 1000 I ran overnight.


# Conclusion, Thoughts, Future Work
**I learned a lot during this project.** <br>
- My eyes were definitely a lot bigger than my stomach in terms of what I thought I could get done
    - A good idea would have been to drop uncommon ingredients from my dataset.
- I wasn't able to use a scoring metric that would have made the results much more useful.
    - A scoring calculation that relates to every other data point in a set is very expensive.
- I ended up using a much too small of an MLP (337 parameters) than was appropriate.
    - Get deeper into typical neural network parameters for a given problem before diving in to training a model over a large number of epochs.
- I was inputting the ingredient combinations to the model as two integers, which didn't give very much real information for the network to train on.
    - One-hot-encoding each ingredient as an input would potentially have given a lot more information.
    
**Future:**
- I'd like to approach the problem again later with a better understanding
- I'd also like to build something on a similar topic using the full [Recipe 1M+](http://pic2recipe.csail.mit.edu/) dataset, which includes all the text in recipes. Predicting quantities or instructions seems very interesting.
- Recreating the KitcheNette project in full seems like it would be a very worthwhile use of time, and I plan on doing the same with other studies I can find that interest me, as it seems like a great way of learning good approaches to problems, and of keeping up to date on what techniques current data science is using.


# References

[KitcheNette (Park et al.)](https://arxiv.org/pdf/1905.07261.pdf)

[Simplified Recipes 1M+](https://dominikschmidt.xyz/simplified-recipes-1M/)

<span>Photo by <a href="https://unsplash.com/@estherwec?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Esther Wechsler</a> on <a href="https://unsplash.com/s/photos/food-pairing?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
