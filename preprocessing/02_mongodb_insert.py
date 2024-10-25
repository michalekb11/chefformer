from model.MongoDBClient import MongoDBClient
import json

mongo = MongoDBClient()

# Read in json recipes
with open('./data/recipes_raw_merged.json') as f:
    recipes = json.load(f)

# Insert transformed data into MongoDB
result = mongo.collection.insert_many(recipes)

# Print inserted IDs
print(f"Inserted {len(result.inserted_ids)} ids.")