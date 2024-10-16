class Recipe:
    def __init__(self, source: str, title: str, ingredients: list[str], instructions: str) -> None:
        self.source = source
        self.title = title
        self.ingredients = ingredients
        self.instructions = instructions

    def to_dict(self):
        return {
            'source':self.source,
            'title':self.title,
            'ingredients':self.ingredients,
            'instructions':self.instructions
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            source=data.get('source', ''),
            title=data.get('title', ''),
            ingredients=data.get('ingredients', []),
            instructions=data.get('instructions')
        )
    
    @classmethod
    def from_mongo(cls, title, db):
        recipe = db.recipes.find_one({"title": title})
        if recipe:
            return cls.from_dict(recipe)
        else:
            raise ValueError(f"Recipe with title '{title}' not found in the database.")
        
    def __str__(self) -> str:
        ingredients_str = '\n'.join([f'- {item}' for item in self.ingredients])
        return f"Title: {self.title}\n\nIngredients:\n{ingredients_str}\n\nInstructions: {self.instructions}"