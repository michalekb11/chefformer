from model.Recipe import Recipe
from model.MongoDBClient import MongoDBClient
import re
import json
from preprocessing.TextCleaner import TextCleaner

#manual_delete_id_set = set([])

# Set up text cleaner
text_cleaner = TextCleaner()

# Connect to mongo
mongo = MongoDBClient()

# Gather all recipes
recipes = [Recipe.from_dict(doc) for doc in mongo.collection.find()]
print(len(recipes))

#====================
for r in recipes:
    # Clean titles
    r.title = text_cleaner.remove_accents(r.title)
    r.title = text_cleaner.remove_special_characters(r.title)
    r.title = text_cleaner.replace_phrases(r.title)
    r.title = text_cleaner.remove_html_tags(r.title)
    r.title = text_cleaner.replace_whitespace(r.title).strip()

    # Clean instructions
    r.ingredients = [
        text_cleaner.remove_advertisement_str(ingred) # Replace 'ADVERTISEMENT', then strip whitespace
        for ingred in r.ingredients
        if text_cleaner.remove_advertisement_str(ingred) # Only keep non-empty strings after processing
    ]
    r.ingredients = [text_cleaner.remove_accents(ingred) for ingred in r.ingredients]
    r.ingredients = [text_cleaner.replace_phrases(ingred) for ingred in r.ingredients]
    r.ingredients = [text_cleaner.remove_html_tags(ingred) for ingred in r.ingredients]
    r.ingredients = [text_cleaner.remove_special_characters(ingred) for ingred in r.ingredients]
    r.ingredients = [text_cleaner.replace_whitespace(ingred).strip() for ingred in r.ingredients]

    # Clean instructions
    r.instructions = text_cleaner.remove_accents(r.instructions)
    r.instructions = text_cleaner.remove_special_characters(r.instructions)
    r.instructions = text_cleaner.replace_phrases(r.instructions)
    r.instructions = text_cleaner.remove_html_tags(r.instructions)
    r.instructions = text_cleaner.replace_whitespace(r.instructions).strip()

#====================
# Item 1: Delete all items that have no title, ingredients, or instructions
# Gather the list indices that meet this criteria
idx_for_delete = [idx for idx, r in enumerate(recipes) if not r.title or not r.ingredients or not r.instructions]
#idx_for_delete.extend([idx for idx, id in enumerate([r.id for r in recipes]) if id in manual_delete_id_set]) # Append some to manually delete
idx_for_delete = list(set(idx_for_delete))

# For each idx, call remove from mongo on that recipe
for idx in idx_for_delete:
    recipes[idx].delete_from_mongo(mongo)

# Pop them from the recipe list
recipes = [r for idx, r in enumerate(recipes) if idx not in idx_for_delete]
print(len(recipes))

#====================
# Item 6: Update mongo with new cleaned recipes
mongo.replace_db(recipes)

# Convert each Recipe object to a dictionary
recipes_json = [recipe.to_dict() for recipe in recipes]

# Save "clean-ish" recipes to file
with open('./data/cleaned/regex_cleaned_recipes.json', 'w') as file:
    json.dump(recipes_json, file, indent=4)




#====================
# Note: I've tried this even after having set up billing. Just runs too slow and costs too much. 
# Will have to do the best we can to clean without using LLM 
# Item 3: Set up LLM to perform recipe cleaning
# gemini = GeminiFlash(prompt_template=FULL_CLEAN_TEMPLATE, temperature=0.1)
# n = len(recipes)

# for idx, r in enumerate(recipes[5000:]):
#     try:
#         new_string = gemini.invoke({'recipe':r.to_string()})
#         new_string = remove_accents(new_string)
#         recipes[idx] = Recipe.from_string(recipe_string=new_string, id=r.id, source=r.source)
#     except Exception as e:
#         print(f'Error occured for idx: {idx}, Id: {r.id}, error: {e}')
#         print()

#     if idx > 0 and (idx % 5000 == 0 or idx == n - 1):
#         save_recipes_to_file(recipes[idx-5000:idx], f"./data/cleaned/llm_cleaned_recipes_{idx}.json")
#         print(f'{round(100 * idx / n, 2)}% completed.')

# Rewrite them all if they all finish.
#save_recipes_to_file(recipes, "./data/cleaned/llm_cleaned_recipes.json")