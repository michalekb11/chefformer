from pymongo import MongoClient

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