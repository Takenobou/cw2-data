# cw2-data

- [x] Task 1
- [x] Task 2 
- [x] Task 3
- [x] Task 4 
- [ ] Task 5
- [ ] Task 6
- [ ] Task 7

## Part I (Guided) – Building up a recommender engine
Your first task is to prepare the data and carry out data pre-processing and visualisation in
order to make sense of the data. Namely, what is the distribution of the ratings; what is the
most liked/popular food recipe and is there any relationship between ratings and number
of ratings. Address the following questions:
1. Load the data; identify and treat possible missing values; show the summary statistics
and the 10 highest rated recipes. [8 marks]
2. Visualise the average ratings and the number of ratings and comment on any
relationship that exist between rating_avg and rating_val. Can you suggest a
threshold
for the number of ratings under which the rating can be considered as not significant.
[8 marks]
Your second task is to actually build up a simple recommender engine.
3. Consider the following selected features:
`
features=['title','rating_avg','rating_val','total_time','category','
cuisine', 'ingredients']
`

    a) Add to your dataframe a column combine_features, which combines all the
contents of the features in the given list features as a single string wherein
each feature’s contents is separated from the other by one space string. [5
marks]

    b) Using the class CountVectorizer and the function cosine_similarity,
compute the cosine similarity matrix from the resulting dataframe of the
combined features. [4 marks]

    c) Consider the recipe ‘Chicken and coconut curry’. Relying on the vector space
method, use a matrix-vector product to show the first 10 recipe
recommendations for a user who has liked that particular recipe. Show the
titles of these recommendations. [10 marks]
## Part II (Open-ended) – Building up and evaluate a recommender engine.
Consider the entire dataset including numerical and categorical features. For example, you
may extend what you did in Part 1.3 by treating the columns rating_av or total_time as
numerical values rather than strings.

4. Write a function, vec_space_method, which takes in a recipe and returns the 10 most
similar recipes to the given one. Do this using a suitable matrix-vector product in the
Vector Space Method. This has to be different from what is carried out in Part 1.3
since you have considered the entire dataset. [15 marks]

5. Write a function, knn_similarity, which takes in a recipe and returns the 10 most similar
recipes to the given one. Do this considering the entire dataset and using the KNN
algorithm. [15 marks]

6. Consider the following test set composed of four users:
   - User 1 likes ‘Chicken tikka masala’
   - User 2 likes ‘Albanian baked lamb with rice’
   - User 3 likes ‘Baked salmon with chorizo rice’
   - User 4 likes ‘Almond lentil stew’
   
    Using this test set, evaluate both recommender systems you have built up in Part 2.4
    and 2.5 in terms of coverage and personalisation. You can add comments on this
    evaluation into your submitted Python code file. [20 marks]

7. Transform the column rating_av of the dataset into a binary format with
    - 1 representing ’tasty’ for a rating average greater than 4.2,
    - -1 representing ’not tasty’, otherwise.
    Build up a model that can predict whether a recipe will be rated as tasty or not. 
    Investigate how accurate is your predictive model. You may consider only average
    ratings that are significant as found in Part 1.2. [15 marks]
