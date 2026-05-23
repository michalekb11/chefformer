from pymongo import MongoClient, UpdateOne
from src.training.database.Recipe import Recipe

class MongoDBClient:
    def __init__(self, uri: str = 'mongodb://localhost:27017', db_name: str = 'chefformer', collection_name: str = 'recipes'):
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print('Connected to MongoDB.')
        except Exception as e:
            print(f'Error connecting to MongoDB: {e}')
            self.client = None
            self.db = None
            self.collection = None

    def update_many(self, recipes: list[Recipe], batch_size=1000):
        total_batches = len(recipes) // batch_size + (1 if len(recipes) % batch_size else 0)
        for i in range(total_batches):
            batch = recipes[i * batch_size: (i + 1) * batch_size]
            operations = [UpdateOne({'id': recipe.id}, {'$set': recipe.to_dict()}) for recipe in batch]
            try:
                self.collection.bulk_write(operations)
            except Exception as e:
                print(f"Error in batch {i}: {e.details}")
            print(f"Batch {i + 1}/{total_batches} update complete.")
        print("All batches completed.")
        return

    def replace_db(self, recipes: list[Recipe]):
        # Clear the mongo db
        delete_result = self.collection.delete_many({})
        print(f"Cleared {delete_result.deleted_count} ids.")

        # Insert transformed data into MongoDB
        insert_result = self.collection.insert_many([r.to_dict() for r in recipes])
        print(f"Inserted {len(insert_result.inserted_ids)} ids.")
    
    def delete_one(self, id: int):
        assert id != -1, "ID is -1. This could delete more than one record."
        result = self.collection.delete_one({'id':id})
        if result.deleted_count <= 0:
            print(f'Something went wrong with the deletion. Deleted count: {result.deleted_count}')
        return