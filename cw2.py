import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Recommender:
    def __init__(self, recipes_path):
        self.df = pd.read_csv(recipes_path)

        features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']
        self.df['combine_features'] = self.df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # Use CountVectorized to convert the text data into a matrix of token counts
        self.vectorizer = CountVectorizer()
        self.features_matrix = self.vectorizer.fit_transform(self.df['combine_features'])

        # Compute the cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.features_matrix)

    def stats(self):
        return self.df.describe()

    def top_rated(self):
        return self.df.sort_values(by='rating_avg', ascending=False).head(10)

    def top(self):
        top10 = self.df.nlargest(10, 'rating_avg')
        return top10[['title', 'rating_avg']]

    def rating_vs_num_ratings(self):
        plt.scatter(self.df['rating_val'], self.df['rating_avg'])
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.title('Relationship between Average Rating and Number of Ratings')
        plt.show()

        threshold = 100  # Define a threshold for the number of ratings

        # Filter the recipes based on the number of ratings threshold
        significant_ratings = self.df[self.df['rating_val'] >= threshold]
        not_significant_ratings = self.df[self.df['rating_val'] < threshold]

        # Plot the significant and not significant ratings separately
        plt.scatter(significant_ratings['rating_val'], significant_ratings['rating_avg'], label='Significant Ratings')
        plt.scatter(not_significant_ratings['rating_val'], not_significant_ratings['rating_avg'],
                    label='Not Significant Ratings')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.title('Relationship between Average Rating and Number of Ratings with Threshold')
        plt.legend()
        plt.show()

    def similar_recipes(self, recipe_index, num_similar=10):
        # Find the indices of the most similar recipes
        similar_indices = self.similarity_matrix[recipe_index].argsort()[::-1][:num_similar + 1]
        similar_indices = similar_indices[similar_indices != recipe_index]

        # Get the titles of the most similar recipes
        similar_recipes = self.df.iloc[similar_indices]['title'].values

        return similar_recipes

    def recommended_recipes(self, recipe_title, num_recommendations=10):
        recipe_index = self.df[self.df['title'] == recipe_title].index[0]
        recipe_vector = self.features_matrix[recipe_index]
        similarity_scores = cosine_similarity(recipe_vector, self.features_matrix)
        similar_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 1]
        recommended_recipes = self.df.iloc[similar_indices]['title'].values
        return recommended_recipes


class Driver:
    def __init__(self, recipes):
        self.recipes = recipes

    def task1(self):
        task1 = Recommender(self.recipes)
        print(task1.stats())
        print(task1.top_rated())
        print(task1.top())
        print(task1.rating_vs_num_ratings())
        print(task1.df['combine_features'])
        print(task1.similar_recipes(0))
        print(task1.recommended_recipes("Chicken and coconut curry"))


def main():
    recipes_path = "recipes.csv"
    driver = Driver(recipes_path)
    driver.task1()


if __name__ == '__main__':
    main()
