import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Recommender:
    def __init__(self, recipes_path):
        self.df = pd.read_csv(recipes_path, na_values=['', ' '])
        self.df.fillna({'category': 'unknown', 'cuisine': 'unknown'}, inplace=True)
        self.df = self.df.drop(['image_url', 'recipe_url'], axis=1)
        self._build_knn_model()

    def _build_knn_model(self):
        # Vectorize recipe titles using TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        title_matrix = self.vectorizer.fit_transform(self.df['title'])

        # Scale numerical features
        self.scaler = MinMaxScaler()
        numerical_features = ['rating_avg', 'rating_val', 'total_time']
        scaled_numerical_matrix = self.scaler.fit_transform(self.df[numerical_features])
        scaled_numerical_df = pd.DataFrame(scaled_numerical_matrix, columns=numerical_features)

        # Combine categorical and numerical features
        self.category_df = pd.get_dummies(self.df['category'])
        self.cuisine_df = pd.get_dummies(self.df['cuisine'])
        combined_matrix = pd.concat(
            [pd.DataFrame(title_matrix.toarray(), columns=self.vectorizer.get_feature_names_out()), scaled_numerical_df,
             self.category_df, self.cuisine_df], axis=1)

        # Convert all column names to strings
        combined_matrix.columns = combined_matrix.columns.astype(str)

        # Build KNN model
        self.knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
        self.knn_model.fit(combined_matrix)

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

    def recommended_recipes(self, recipe_title):
        df = self.df.copy()
        num_recommendations = 10
        features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']

        df['combine_features'] = df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # Use CountVectorizer to convert the text data into a matrix of token counts
        vectorizer = CountVectorizer()
        features_matrix = vectorizer.fit_transform(df['combine_features'])

        # Compute the cosine similarity matrix
        similarity_matrix = cosine_similarity(features_matrix)

        recipe_index = df[df['title'] == recipe_title].index[0]

        # Get the indices of the most similar recipes
        similar_indices = similarity_matrix[recipe_index].argsort()[::-1][1:num_recommendations + 1]

        # Get the titles of the most similar recipes
        recommended_recipes = df.iloc[similar_indices]['title'].to_string()
        return recommended_recipes

    def vec_space_method(self, recipe):
        # Preprocess the dataset
        df = self.df.copy()
        # One-hot encode categorical features
        categorical_features = ['category', 'cuisine', 'ingredients']
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_features]).toarray()

        # Normalize numerical features
        numerical_features = ['rating_avg', 'rating_val', 'total_time']
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(df[numerical_features])

        # Combine one-hot encoded and normalized features
        processed_data = np.hstack((one_hot_encoded, normalized))

        # Find the index of the given recipe in the dataset
        recipe_index = df[df['title'] == recipe].index[0]

        # Compute cosine similarity between the given recipe and all other recipes
        similarities = cosine_similarity(processed_data[recipe_index].reshape(1, -1), processed_data)

        # Find the indices of the 10 most similar recipes
        most_similar_indices = np.argsort(similarities[0])[-11:-1][::-1]

        # Return the 10 most similar recipes
        return df.iloc[most_similar_indices]['title'].to_string()

    def knn_similarity(self, title):
        # Transform the title using the fitted vectorizer
        title_vector = self.vectorizer.transform([title])

        # Get numerical features
        numerical_features = ['rating_avg', 'rating_val', 'total_time']
        input_numerical_vector = self.df[self.df['title'] == title][numerical_features]

        # Scale numerical features
        if not input_numerical_vector.empty:
            scaled_numerical_vector = self.scaler.transform(input_numerical_vector)
        else:
            scaled_numerical_vector = np.zeros((1, len(numerical_features)))

        # Get categorical features
        input_category_vector = self.category_df[self.df['title'] == title]
        if input_category_vector.empty:
            input_category_vector = np.zeros((1, self.category_df.shape[1]))

        input_cuisine_vector = self.cuisine_df[self.df['title'] == title]
        if input_cuisine_vector.empty:
            input_cuisine_vector = np.zeros((1, self.cuisine_df.shape[1]))

        # Combine the transformed features
        input_vector = pd.concat([pd.DataFrame(title_vector.toarray(), columns=self.vectorizer.get_feature_names_out()), pd.DataFrame(scaled_numerical_vector, columns=numerical_features), pd.DataFrame(input_category_vector.values, columns=self.category_df.columns), pd.DataFrame(input_cuisine_vector.values, columns=self.cuisine_df.columns)], axis=1)

        # Convert all column names to strings
        input_vector.columns = input_vector.columns.astype(str)

        # Get 10 most similar recipe indices and distances
        distances, indices = self.knn_model.kneighbors(input_vector, return_distance=True)

        # Return the most similar recipe titles, excluding the first one (the same title)
        return self.df.iloc[indices[0][1:]]['title'].to_string()


class Driver:
    def __init__(self, recipes):
        self.recipes = recipes
        self.df = Recommender(self.recipes)

    def task1(self):
        task1 = self.df
        print("=" * 50, "Task 1", "=" * 50)
        print(task1.stats())
        print(task1.top_rated())

    def task2(self):
        task2 = self.df
        print("=" * 50, "Task 2", "=" * 50)
        print(task2.top())
        print(task2.rating_vs_num_ratings())

    def task3(self):
        task3 = self.df
        print("=" * 50, "Task 3", "=" * 50)
        print(task3.recommended_recipes("Chicken and coconut curry"))

    def task4(self):
        task4 = self.df
        print("=" * 50, "Task 4", "=" * 50)
        print(task4.vec_space_method("Chicken and coconut curry"))

    def task5(self):
        task5 = self.df
        print("=" * 50, "Task 5", "=" * 50)
        print(task5.knn_similarity("Chicken and coconut curry"))


def main():
    recipes_path = "recipes.csv"
    driver = Driver(recipes_path)
    driver.task1()
    driver.task2()
    driver.task3()
    driver.task4()
    driver.task5()


if __name__ == '__main__':
    main()
