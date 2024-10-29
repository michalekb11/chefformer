from pymongo import MongoClient, UpdateOne
from model.Recipe import Recipe

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

    def update_many(self, recipes: list[Recipe]):
        operations = [UpdateOne({'id':recipe.to_dict()['id']}, {'$set':recipe.to_dict()}) for recipe in recipes]
        self.collection.bulk_write(operations)
        print('Batch update complete.')
        return
    
    def delete_one(self, id: int):
        assert id != -1, "ID is -1. This could delete more than one record."
        result = self.collection.delete_one({'id':id})
        if result.deleted_count <= 0:
            print(f'Something went wrong with the deletion. Deleted count: {result.deleted_count}')
        return