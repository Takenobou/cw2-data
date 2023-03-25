import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Recommender:
    def __init__(self, recipes_path):
        """
            Initialize the Recommender class with a path to the recipes dataset.

            :param str recipes_path: A string representing the path to a CSV file containing recipe data.
                    The file should contain a header row with column names and the first column should be
                    an index or ID column. If any values in the 'category' or 'cuisine' columns are missing,
                    they will be filled with the string 'unknown'. Any other missing values in the dataset should
                    be represented as NaN.
        """

        # Load the dataset into a pandas DataFrame
        self.df = pd.read_csv(recipes_path, na_values=['', ' '])

        # Fill any missing values in the 'category' and 'cuisine' columns with 'unknown'
        self.df.fillna({'category': 'unknown', 'cuisine': 'unknown'}, inplace=True)

        # Drop the 'image_url' and 'recipe_url' columns, which are not needed for recommendation
        self.df = self.df.drop(['image_url', 'recipe_url'], axis=1)

        # Drop the first column of the dataset, which is assumed to be an index or ID column
        self.df = self.df.drop(self.df.columns[0], axis=1)

        # Build a K-Nearest Neighbors model based on the recipe dataset
        self._build_knn_model()

    # Part 1 Task 1
    def stats(self):
        """
            Return summary statistics for the recipe dataset.

            :return pd.DataFrame: A pandas DataFrame containing summary statistics for the recipe dataset.
        """
        return self.df.describe()

    # Part 1 Task 1
    def top(self):
        """
            Return the top 10 recipes with the highest average rating.

            :return pd.DataFrame: A pandas DataFrame containing the top 10 recipes with the highest average rating,
                including the 'title' and 'rating_avg' columns.
        """
        top10 = self.df.nlargest(10, 'rating_avg')
        top10 = top10[['title', 'rating_avg']].sort_values('rating_avg', ascending=False).reset_index(drop=True)
        top10.columns = ['Recipe Title', 'Average Rating']
        return top10

    # Part 1 Task 2
    def rating_vs_num_ratings(self):
        """
            Generate scatter plots of average rating vs. number of ratings for all recipes,
            and for recipes with and without significant ratings.

            :return str: A message indicating that the graphs have been generated.
        """

        # Generate a scatter plot of average rating vs. number of ratings for all recipes
        plt.scatter(self.df['rating_val'], self.df['rating_avg'])
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.title('Relationship between Average Rating and Number of Ratings')
        plt.show()

        # Define a threshold for the number of ratings, this chooses the top 25% of recipes with the most ratings
        threshold = self.df['rating_val'].quantile(0.75)

        # Filter the recipes based on the number of ratings threshold
        significant_ratings = self.df[self.df['rating_val'] >= threshold]
        not_significant_ratings = self.df[self.df['rating_val'] < threshold]

        # Generate a scatter plot of average rating vs. number of ratings
        # for recipes with and without significant ratings
        plt.scatter(significant_ratings['rating_val'], significant_ratings['rating_avg'], label='Significant Ratings')
        plt.scatter(not_significant_ratings['rating_val'], not_significant_ratings['rating_avg'],
                    label='Not Significant Ratings')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.title('Relationship between Average Rating and Number of Ratings with Threshold')
        plt.legend()
        plt.show()

        # Return a message indicating that the graphs have been generated
        return f"Scatter plots of average rating vs. number of ratings have been generated."

    # Part 1 Task 3
    def recommended_recipes(self, recipe_title):
        """
            Generate recommendations for a given recipe title using cosine similarity.

            :param str recipe_title: The title of the recipe for which to generate recommendations.

            :return pd.Series: A pandas Series containing the titles of the 10 most similar recipes to the input recipe,
                    in descending order of similarity.
        """
        # Make a copy of the dataset
        df = self.df.copy()

        # Define the number of recommended recipes to return
        num_recommendations = 10

        # Define the features to use for similarity calculation
        features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']

        # Combine the selected features into a single text feature
        df['combine_features'] = df[features].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # Use CountVectorizer to convert the text data into a matrix of token counts
        vectoriser = CountVectorizer()
        features_matrix = vectoriser.fit_transform(df['combine_features'])

        # Compute the cosine similarity matrix
        similarity_matrix = cosine_similarity(features_matrix)

        # Find the index of the recipe with the given title
        recipe_index = df[df['title'] == recipe_title].index[0]

        # Get the indices of the most similar recipes
        similar_indices = similarity_matrix[recipe_index].argsort()[::-1][1:num_recommendations + 1]

        # Return the recommended recipe titles
        return df.iloc[similar_indices]['title']

    # Part 2 Task 1
    def vec_space_method(self, recipe_title):
        """
            Generate recommendations for a given recipe title using the vector space method.

            :param str recipe_title: The title of the recipe for which to generate recommendations.

            :return pd.Series: A pandas Series containing the titles of the 10 most similar recipes to the input recipe,
                in descending order of similarity.
        """

        # Make a copy of the dataset
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
        recipe_index = df[df['title'].str.contains(recipe_title, case=False)].index[0]

        # Compute cosine similarity between the given recipe and all other recipes
        similarities = cosine_similarity(processed_data[recipe_index].reshape(1, -1), processed_data)

        # Find the indices of the 10 most similar recipes
        most_similar_indices = np.argsort(similarities[0])[-11:-1][::-1]

        # Return the 10 most similar recipes
        return df.iloc[most_similar_indices]['title']

    # Part 2 Task 2
    def _build_knn_model(self):
        """
            Build a KNN model for recipe recommendations.

            This method uses the KNN algorithm to find the k most similar recipes to a given recipe.

            The model is built by vectorising the recipe titles using TF-IDF, scaling numerical features,
            one-hot encoding categorical features, and combining all features into a single matrix.

            The KNN model is trained on the combined feature matrix, using cosine similarity as the distance metric.
        """

        # Vectorize recipe titles using TF-IDF
        self.vectoriser = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        title_matrix = self.vectoriser.fit_transform(self.df['title'])

        # Scale numerical features
        self.scaler = MinMaxScaler()
        numerical_features = ['rating_avg', 'rating_val', 'total_time']
        scaled_numerical_matrix = self.scaler.fit_transform(self.df[numerical_features])
        scaled_numerical_df = pd.DataFrame(scaled_numerical_matrix, columns=numerical_features)

        # One-hot encode categorical features
        self.category_df = pd.get_dummies(self.df['category'])
        self.cuisine_df = pd.get_dummies(self.df['cuisine'])

        # Combine all features into a single matrix
        combined_matrix = pd.concat(
            [pd.DataFrame(title_matrix.toarray(), columns=self.vectoriser.get_feature_names_out()), scaled_numerical_df,
             self.category_df, self.cuisine_df], axis=1)

        # Convert all column names to strings
        combined_matrix.columns = combined_matrix.columns.astype(str)

        # Build KNN model
        self.knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
        self.knn_model.fit(combined_matrix)

    # Part 2 Task 2
    def knn_similarity(self, recipe_title):
        """
        Returns a pandas Series containing the titles of the 10 most similar recipes to the input recipe.

        :param str recipe_title: The title of the recipe for which to generate recommendations.

        :return pd.Series: A pandas Series containing the titles of the 10 most similar recipes to the input recipe,
            excluding the input recipe title.
        """

        # Transform the title using the fitted vectoriser
        title_vector = self.vectoriser.transform([recipe_title])

        # Get numerical and categorical features in one pass
        input_row = self.df.query("title == @recipe_title")

        # Scale numerical features
        numerical_features = ['rating_avg', 'rating_val', 'total_time']
        if not input_row.empty:
            scaled_numerical_vector = self.scaler.transform(input_row[numerical_features])
        else:
            scaled_numerical_vector = np.zeros((1, len(numerical_features)))

        # Get categorical features
        input_category_vector = self.category_df.loc[input_row.index].to_numpy()
        if input_category_vector.size == 0:
            input_category_vector = np.zeros((1, self.category_df.shape[1]))

        input_cuisine_vector = self.cuisine_df.loc[input_row.index].to_numpy()
        if input_cuisine_vector.size == 0:
            input_cuisine_vector = np.zeros((1, self.cuisine_df.shape[1]))

        # Combine the transformed features
        input_vector = pd.concat([pd.DataFrame(title_vector.toarray(), columns=self.vectoriser.get_feature_names_out()),
                                  pd.DataFrame(scaled_numerical_vector, columns=numerical_features),
                                  pd.DataFrame(input_category_vector, columns=self.category_df.columns),
                                  pd.DataFrame(input_cuisine_vector, columns=self.cuisine_df.columns)], axis=1)

        # Convert all column names to strings
        input_vector.columns = input_vector.columns.astype(str)

        # Get 10 most similar recipe indices and distances
        distances, indices = self.knn_model.kneighbors(input_vector, return_distance=True)

        # Return the most similar recipe titles, excluding the input recipe title
        return self.df.iloc[indices[0][1:]]['title']

    # Part 2 Task 3
    def calculate_metrics(self, recommendations):
        """
        Calculates the coverage and personalisation metrics for a set of recommendations.

        :param (Dict[int, List[str]] recommendations: A dictionary mapping user IDs to recommended recipe title lists.

        :return Tuple[float, float]: A tuple containing the coverage and personalisation metrics as floats.
        """

        all_recommendations = set()
        total_items = self.df.shape[0]
        unique_recommendations = 0
        cosine_similarities = []

        # Calculate the number of unique recommended items and the total number of recommended items
        for user_recommendations in recommendations.values():
            all_recommendations.update(user_recommendations)
            unique_recommendations += len(set(user_recommendations))

        # Calculate the coverage metric
        coverage = len(all_recommendations) / total_items

        recommendation_vectors = []
        for recs in recommendations.values():
            # Create a binary vector of recommended items for each user
            rec_vector = [1 if r in recs else 0 for r in all_recommendations]
            recommendation_vectors.append(rec_vector)

        # Calculate the cosine similarities between the user vectors
        cosine_similarities_matrix = cosine_similarity(recommendation_vectors)

        for i in range(cosine_similarities_matrix.shape[0]):
            for j in range(i + 1, cosine_similarities_matrix.shape[1]):
                cosine_similarities.append(cosine_similarities_matrix[i, j])

        # Calculate the personalisation metric
        personalisation = 1 - np.mean(cosine_similarities)

        return coverage, personalisation

    # Part 2 Task 3
    def evaluate_recommenders(self, test_set):
        """
        Evaluates the KNN and vector space recommenders on a test set of user-liked recipes.

        :param Dict[int, str] test_set : A dictionary mapping user IDs to recipe titles that they liked.

        :return str: A string summarizing the evaluation results for the KNN and vector space recommenders.
        """

        knn_recommendations = {}
        vec_space_recommendations = {}

        # Generate recommendations for each user in the test set using both recommenders
        for user, liked_recipe in test_set.items():
            knn_recommendations[user] = self.knn_similarity(liked_recipe).tolist()
            vec_space_recommendations[user] = self.vec_space_method(liked_recipe).tolist()

        # Calculate the coverage and personalisation metrics for both recommenders
        knn_coverage, knn_personalisation = self.calculate_metrics(knn_recommendations)
        vec_space_coverage, vec_space_personalisation = self.calculate_metrics(vec_space_recommendations)

        # Format the evaluation results into a string
        result = f"Based on a test set of {len(test_set)} users:\n"
        result += f"KNN Recommender: Coverage = {knn_coverage:.3f}," \
                  f" Personalisation = {knn_personalisation:.2f}\n"
        result += f"Vector Space Recommender: Coverage = {vec_space_coverage:.3f}, " \
                  f"Personalisation = {vec_space_personalisation:.2f}"
        return result

    def tasty_check(self, recipe_title):
        """
            Checks if a given recipe is "tasty" (i.e., has an average rating above 4.2)
             using a logistic regression model.

            :param str recipe_title: The title of the recipe to check.

            :return str: A string indicating whether the input recipe is tasty or not,
                 and the accuracy of the logistic regression model.
        """

        # Transform the rating_avg column into binary format
        self.df['tasty'] = np.where(self.df['rating_avg'] > 4.2, 1, -1)

        # Consider only significant average ratings
        significant_ratings = self.df[self.df['rating_val'] >= self.df['rating_val'].quantile(0.75)]

        # Prepare the data for training
        x = significant_ratings.drop(['tasty', 'title'], axis=1)
        y = significant_ratings['tasty']

        # One-hot encode the categorical variables
        x = pd.get_dummies(x)

        # Split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Build the logistic regression model
        model = LogisticRegression(max_iter=5000)
        model.fit(x_train, y_train)

        # Investigate the accuracy of the model
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Check if the input dish is tasty or not
        input_recipe = self.df[self.df['title'] == recipe_title]
        if not input_recipe.empty:
            input_recipe_encoded = pd.get_dummies(input_recipe.drop(['tasty', 'title'], axis=1))
            input_recipe_encoded = input_recipe_encoded.reindex(columns=x.columns, fill_value=0)
            tasty_pred = model.predict(input_recipe_encoded)
            tasty = "tasty" if tasty_pred[0] == 1 else "not tasty"
        else:
            tasty = "unknown as it has not been rated enough"

        return f"The input dish '{recipe_title}' is {tasty}. The model's accuracy is {accuracy:.2f}."


class Driver:
    def __init__(self, recipes_path):
        """
        Constructor for the Driver class.
        :param str recipes_path: The path to the CSV file containing recipe data.
        """
        self.recipes = recipes_path
        self.df = Recommender(self.recipes)

    def task1(self):
        """
        Print statistics for the recipe dataset and the top 10 highest rated recipes.
        """
        task1 = self.df
        print("=" * 50, "Task 1", "=" * 50)
        print(task1.stats())
        print(task1.top())

    def task2(self):
        """
        Plot the relationship between average rating and number of ratings for all recipes, and
        separately for recipes with a number of ratings greater than or equal to 100.
        """
        task2 = self.df
        print("=" * 50, "Task 2", "=" * 50)
        print(task2.rating_vs_num_ratings())

    def task3(self, recipe_title):
        """
        Print the top 10 recommended recipes based on cosine similarity with the input recipe.
        :param str recipe_title: The title of the recipe to use as the basis for recommendations.
        """
        task3 = self.df
        print("=" * 50, "Task 3", "=" * 50)
        recommended_recipes = '\n'.join(task3.recommended_recipes(recipe_title))
        print(f"Recommended recipes for {recipe_title}:\n\n{recommended_recipes}")

    def task4(self, recipe_title):
        """
        Print the top 10 recommended recipes based on the vector space model method with the input recipe.
        :param str recipe_title: The title of the recipe to use as the basis for recommendations.
        """
        task4 = self.df
        print("=" * 50, "Task 4", "=" * 50)
        # Return the recommended recipe titles as a string
        recommended_recipes = '\n'.join(task4.vec_space_method(recipe_title).to_list())
        print(f"Recommended recipes for {recipe_title}:\n\n{recommended_recipes}")

    def task5(self, recipe_title):
        """
        Print the titles of the 10 most similar recipes to the input recipe based on the KNN method.
        :param str recipe_title: The title of the recipe to use as the basis for similarity comparisons.
        """
        task5 = self.df
        print("=" * 50, "Task 5", "=" * 50)
        recommended_recipes = '\n'.join(task5.knn_similarity(recipe_title).to_list())
        print(f"Recommended recipes for {recipe_title}:\n\n{recommended_recipes}")

    def task6(self, test_set):
        """
        Evaluate the KNN and Vector Space Recommender systems on a test set.
        :param dict test_set: A dictionary of test users and their respective recipe ratings.
        """
        task6 = self.df
        print("=" * 50, "Task 6", "=" * 50)
        # Based on the evaluated metrics, the KNN Recommender outperforms the Vector Space Recommender
        # in terms of both coverage and personalization. The KNN Recommender offers a wider range of
        # recommendations and better tailors suggestions to individual users. However, these conclusions
        # are drawn from a small test set of 4 users, and the performance may vary with larger, more
        # diverse test sets.
        print(task6.evaluate_recommenders(test_set))

    def task7(self, recipe_title):
        """
        Check the tastiness of a given recipe.
        :param str recipe_title: The title of the recipe to check.
        """
        task7 = self.df
        print("=" * 50, "Task 7", "=" * 50)
        print(task7.tasty_check(recipe_title))


def main():
    # Test data
    recipes_path = "recipes.csv"
    # Test recipe
    test_recipe = "Chicken and coconut curry"
    # Test set for task 6
    test_set = {
        'User 1': 'Chicken tikka masala',
        'User 2': 'Albanian baked lamb with rice',
        'User 3': 'Baked salmon with chorizo rice',
        'User 4': 'Almond lentil stew'
    }

    # Run the driver
    driver = Driver(recipes_path)
    # Part 1
    driver.task1()
    driver.task2()
    driver.task3(test_recipe)
    # Part 2
    driver.task4(test_recipe)
    driver.task5(test_recipe)
    driver.task6(test_set)
    driver.task7(test_recipe)


if __name__ == '__main__':
    main()
