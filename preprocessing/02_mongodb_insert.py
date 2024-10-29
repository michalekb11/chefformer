from model.MongoDBClient import MongoDBClient
import json

mongo = MongoDBClient()

# Read in json recipes
with open('./data/recipes_raw_merged.json') as f:
    recipes = json.load(f)

# Clear the mongo db
delete_result = mongo.collection.delete_many({})
print(f"Cleared {delete_result.deleted_count} ids.")

# Insert transformed data into MongoDB
insert_result = mongo.collection.insert_many(recipes)
print(f"Inserted {len(insert_result.inserted_ids)} ids.")