## [Simplified Recipes Dataset](#https://dominikschmidt.xyz/simplified-recipes-1M/)

Alternative datasets: <br>
[Food.com recipes and user interactions](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) <br>
[Epicurious rceipes](https://www.kaggle.com/hugodarwood/epirecipes) <br>



### Data Structure
**2 Numpy Arrays**
- ingredients: numpy array of 3500 ingredients as strings
- recipes: numpy array of 1067557 recipes as lists of ingredient indices


# Goals
### MVP
- ~~(original) given an ingredient, predict the rest of ingredients in a recipe~~
- ~~realistically, predict top _n_ most likely recipes~~
- predict rating of flavour pairings
### More
- remove problem ingredients
- remove things like seasoning
- predict recipes _n_ ingredients
- measure of novelty (derived distance function?)
- input more (1-3) ingredients

- I really like the visualizations [here](#https://pudding.cool/2018/05/cookies/). a simplified (non animated probably) version of this type of thing would be a cool addition if extra time
