import re

class Recipe:
    def __init__(self, id: int, source: str, title: str, ingredients: list[str], instructions: str) -> None:
        self.id = id or -1
        self.source = source or ''
        self.title = title or ''
        self.ingredients = ingredients or []
        self.instructions = instructions or ''

    def to_dict(self):
        return {
            'id':self.id,
            'source':self.source,
            'title':self.title,
            'ingredients':self.ingredients,
            'instructions':self.instructions
        }
    
    def to_string(self, include_id=False, include_source=False):
        ingredients_str = '\n'.join([f'- {item}' for item in self.ingredients])
        final_string = f"Title: {self.title}\n\nIngredients:\n{ingredients_str}\n\nInstructions: {self.instructions}"
        if include_source:
            final_string = f'Source: {self.source}\n\n' + final_string
        if include_id:
            final_string = f'Id: {self.id}\n\n' + final_string
        return final_string
    
    def delete_from_mongo(self, mongo):
        mongo.delete_one(self.id)
        return
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data.get('id', -1),
            source=data.get('source', ''),
            title=data.get('title', ''),
            ingredients=data.get('ingredients', []),
            instructions=data.get('instructions', '')
        )
    
    @classmethod
    def from_mongo(cls, id, db):
        recipe = db.recipes.find_one({"id": id})
        if recipe:
            return cls.from_dict(recipe)
        else:
            raise ValueError(f"Recipe with id '{id}' not found in the database.")
        
    @classmethod
    def from_string(cls, recipe_string, id=-1, source=''):
        # Extract the title
        title_match = re.search(r"Title:\s*(.+)", recipe_string)
        title = title_match.group(1).strip() if title_match else ''

        # Extract the ingredients
        ingredients_match = re.search(r"Ingredients:\s*((?:- .+\n?)+)", recipe_string)
        ingredients = ingredients_match.group(1).strip().splitlines() if ingredients_match else []
        ingredients = [ingredient[2:].strip() for ingredient in ingredients]  # Remove "- " from each line

        # Extract the instructions
        instructions_match = re.search(r"Instructions:\s*(.+)", recipe_string, re.DOTALL)
        instructions = instructions_match.group(1).strip() if instructions_match else ''

        # Return an instance of Recipe
        return cls(id=id, source=source, title=title, ingredients=ingredients, instructions=instructions)
        
    def __str__(self) -> str:
        ingredients_str = '\n'.join([f'- {item}' for item in self.ingredients])
        return f"Id: {self.id}\n\nSource: {self.source}\n\nTitle: {self.title}\n\nIngredients:\n{ingredients_str}\n\nInstructions: {self.instructions}"
