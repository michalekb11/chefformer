from model.Recipe import Recipe
from model.MongoDBClient import MongoDBClient
from model.GeminiFlash import GeminiFlash
import re
from preprocessing.prompts.CleaningPromptTemplates import TITLE_CLEAN_TEMPLATE, INSTRUCTION_CLEAN_TEMPLATE

# Connect to mongo
mongo = MongoDBClient()

# Gather all recipes
recipes = [Recipe.from_dict(doc) for doc in mongo.collection.find()]

#====================
# Item 1: Delete all items that have no title, ingredients, or instructions
# Gather the list indices that meet this criteria
idx_for_delete = [idx for idx, r in enumerate(recipes) if not r.title and not r.ingredients and not r.instructions]

# For each idx, call remove from mongo on that recipe
for idx in idx_for_delete:
    recipes[idx].delete_from_mongo(mongo)

# Pop them from the recipe list
recipes = [r for idx, r in enumerate(recipes) if idx not in idx_for_delete]

#====================
# Item 2: Remove 'ADVERTISEMENT' from AR ingredients
# The LLM would probably do this for us, but doing it here is easy and gives one less task to the LLM
for r in recipes:
    r.ingredients = [
        ingred.replace('ADVERTISEMENT', '').strip() # Replace 'ADVERTISEMENT', then strip whitespace
        for ingred in r.ingredients
        if ingred.replace('ADVERTISEMENT', '').strip() # Only keep non-empty strings after processing
    ]

#====================
# Item 3: Set up LLM to perform recipe cleaning
title_gemini = GeminiFlash(prompt_template=TITLE_CLEAN_TEMPLATE, temperature=0.1)
instruction_gemini = GeminiFlash(prompt_template=INSTRUCTION_CLEAN_TEMPLATE, temperature=0.1)
for r in recipes:
    r.title = title_gemini.invoke({'recipe':r.to_string()})
    r.instructions = instruction_gemini.invoke({'recipe':r.to_string()})


#====================
# Item 4: Reparse the recipes to identify title, list of ingredients, and instructions




#====================
# Item 5: Final clean of text: strip whitespace, convert to lowercase if necessary, remove all special characters from title, ingredients, instructions
for r in recipes:
    r.title = re.sub(r"[^a-zA-Z0-9 .,/:;\'\"[\]{}+=\-_–()*&^%$#@!~\\|°<>?]", '', r.title).replace('\n', '').replace('\t', '').strip() # Keep all standard keyboard characters
    r.ingredients = [re.sub(r"[^a-zA-Z0-9 .,/:;\'\"[\]{}+=\-_–()*&^%$#@!~\\|°<>?]", '', ingred).replace('\n', '').replace('\t', '').strip() for ingred in r.ingredients]
    r.instructions = re.sub(r"[^a-zA-Z0-9 .,/:;\'\"[\]{}+=\-_–()*&^%$#@!~\\|°<>?]", '', r.instructions).replace('\n', '').replace('\t', '').strip()

#====================
# Item 6: Update mongo with new cleaned recipes
mongo.update_many(recipes)