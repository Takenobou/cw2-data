import pandas as pd


class Recommender:
    def __init__(self, recipes_path):
        self.df = pd.read_csv(recipes_path)

    def stats(self):
        return self.df.describe()

    def top_rated(self):
        return self.df.sort_values(by='rating_avg', ascending=False).head(10)


class Driver:
    def __init__(self, recipes):
        self.recipes = recipes

    def task1(self):
        task1 = Recommender(self.recipes)
        print(task1.stats())
        print(task1.top_rated())


def main():
    recipes_path = "recipes.csv"
    driver = Driver(recipes_path)
    driver.task1()


if __name__ == '__main__':
    main()
