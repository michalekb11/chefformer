import json

def remove_picture_link(data):
    """Removing the picture_link key to clean up JSONs"""
    for key in list(data.keys()):
        if isinstance(data[key], dict) and "picture_link" in data[key]:
            del data[key]["picture_link"]
    return

def add_source(data, source):
    for key, recipe_dict in data.items():
        recipe_dict['source'] = source
    return
#===========================================
# Import the raw data
with open('data/recipes_raw_nosource_ar.json') as file:
    ar = json.load(file)

with open('data/recipes_raw_nosource_fn.json', 'r') as file:
    fn = json.load(file)

with open('data/recipes_raw_nosource_epi.json') as file:
    epi = json.load(file)


# Remove picture link keys
remove_picture_link(ar)
remove_picture_link(epi)
remove_picture_link(fn)

# Add the source
add_source(ar, 'ar')
add_source(epi, 'epi')
add_source(fn, 'fn')

# Merge JSONs together
merged_raw = {**ar, **epi, **fn}

# Transform data into the desired format
transformed_data = []
for idx, (key, recipe) in enumerate(merged_raw.items(), start=1):
    new_recipe = {
        "id": idx,
        "source": recipe.get("source", ''),
        "title": recipe.get("title", ''),
        "ingredients": recipe.get("ingredients", []),
        "instructions": recipe.get("instructions", '')
    }
    transformed_data.append(new_recipe)

# Create new json
with open('data/recipes_raw_merged.json', 'w') as f:
    json.dump(transformed_data, f, indent=4)
