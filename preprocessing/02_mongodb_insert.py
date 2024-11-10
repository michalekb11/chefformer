from database.MongoDBClient import MongoDBClient
import json

mongo = MongoDBClient()

# Read in json recipes
with open('./data/recipes_raw_merged.json') as f:
    recipes = json.load(f)

mongo.replace_db(recipes)