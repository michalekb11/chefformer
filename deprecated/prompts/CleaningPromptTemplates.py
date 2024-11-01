from langchain_core.prompts import PromptTemplate

#=====Clean entire recipe at once======
full_clean_prompt = """You are an expert in writing cooking recipes. Please clean and standardize the following recipe according to the format provided below. Your response should be organized with a title, ingredients, and instructions, and each section should be professionally formatted without extraneous information. Follow these instructions closely (instructions provided for each section of the recipe). 

Title:
- Ensure the title is short, clean, and descriptive, focusing only on the food or dish itself.
- Remove unnecessary adjectives, personal names, spelling errors, or punctuation that doesn't contribute to clarity. Do not assume or add details not present in the recipe.
- If the dish type isn't clear from the recipe, leave the title blank (empty string).

Ingredients:
- List each ingredient on its own line, prefixed by a hyphen -, ensuring quantities (if available) are accurate and listed alongside the ingredients.
- If a line contains more than one ingredient, separate them as individual entries where appropriate.
- If no ingredients are provided, infer an ingredient list only if possible based on the title and instructions; otherwise, leave this section blank (empty string).

Instructions:
- Write the instructions in a clear, professional paragraph format, summarizing the actions needed to make the recipe.
- Remove any duplicate steps, extraneous text, or promotional information. Ensure only direct, actionable steps are included.
- If the instructions are unclear or incomplete due to missing ingredients or lack of a discernible dish, leave this section blank (empty string).
- Remove any HTML tags or other non-recipe text to maintain clarity.

Strictly follow the output format below. This is a crucial part of the cleaning process. The recipe MUST follow this exact format.
<start of output format>Title: (recipe title)

Ingredients:
- (each ingredient on a new line with quantity if available)

Instructions: (instructions in one paragraph, each step clearly presented and duplicates removed)<end of output format>

Now, you will see 5 examples of messy recipes and the corresponding cleaned recipes following the output format above. Using these examples, you should learn how to clean the recipes properly (focus on format, what information is included / rerewritten, what information is removed, etc.).

Example 1
Title: Strawberry Prosecco Soup 

Ingredients:
- strawberries with Prosecco, tarragon, salt, and 2 tablespoons sugar
- Purée mixture in a blender until smooth, then set aside 1 cup purée. Blend remaining mixture with yogurt and sugar to taste. Serve soup drizzled with reserved purée.

Instructions: 

Example 1 cleaned
Title: Strawberry Prosecco Soup

Ingredients:
- Strawberries
- Prosecco
- Tarragon
- Salt
- Sugar (2 tablespoons)
- Yogurt (to taste)

Instructions: Purée strawberries with Prosecco, tarragon, salt, and 2 tablespoons of sugar in a blender until smooth. Set aside 1 cup of purée. Blend the remaining mixture with yogurt and sugar to taste. Serve the soup drizzled with reserved purée.

Example 2
Title: Dinner Party Recipes

Ingredients:


Instructions: 

Example 2 cleaned
Title:

Ingredients:

Instructions:

Example 3
Title: To Clean Calf Fries 

Ingredients:
- Animal fries, or testicles, have a mild flavor and delicate texture somewhat like sweetbreads. They are considered a seasonal delicacy in many parts of the world. Gourmet first wrote about them in 1942, as a treat that hungry cowboys cooked for themselves.
- Calf fires are available frozen from Highland Country Farms (800-398-7803).
- Cleaning fries (once they've been thawed) can be a little unsettling —but if you think about what you are doing, so can the preparation of any creature. Each fry is enclosed in a sac of skin with a small opening. Pull the opening apart with your fingers to reveal the tender, membrane-covered portion inside. Cut the skin away with a knife, leaving the membrane intact, then proceed with the recipe.

Instructions: 

Example 2 cleaned
Title: To Clean Calf Fries

Ingredients:
- Calf fries (testicles) with a mild flavor similar to sweetbreads
- Frozen calf fries, available from Highland Country Farms

Instructions: Once thawed, each fry is enclosed in a sac of skin with a small opening. Pull the opening apart with your fingers to reveal the tender, membrane-covered portion inside. Cut the skin away with a knife, leaving the membrane intact. Proceed with the recipe of your choice.

Example 4
Title: Almond-Chocolate Macaroons 

Ingredients:
- 2 cups (about 9 1/2 ounces) whole almonds, toasted
- 1 cup sugar
- 1/4 teaspoon ground cinnamon
- 1/8 teaspoon salt
- 1 large egg
- 2 large egg white
- 1/4 teaspoon almond extract
- 3/4 cup finely chopped bittersweet (not unsweetened) or semisweet chocolate

Instructions: Position rack in center of oven and preheat to 350°F. Line 2 heavy large baking sheets with foil. Butter foil and dust with flour. Finely grind almonds, sugar, cinnamon and salt in processor. Add egg, egg white and almond extract and process until mixture holds together. Transfer to bowl. Stir in chocolate. Using moistened hands, roll mixture into 1-inch balls and place on prepared sheets. Flatten to 1/3-inch thick rounds. Bake macaroons until tops puff and centers are still soft, about 12 minutes. Transfer to rack and cool completely. Store in airtight container at room temperature. (Can be prepared 3 days ahead.)
Position rack in center of oven and preheat to 350°F. Line 2 heavy large baking sheets with foil. Butter foil and dust with flour. Finely grind almonds, sugar, cinnamon and salt in processor. Add egg, egg white and almond extract and process until mixture holds together. Transfer to bowl. Stir in chocolate.
Using moistened hands, roll mixture into 1-inch balls and place on prepared sheets. Flatten to 1/3-inch thick rounds. Bake macaroons until tops puff and centers are still soft, about 12 minutes. Transfer to rack and cool completely. Store in airtight container at room temperature. (Can be prepared 3 days ahead.)

Example 4 cleaned
Title: Almond-Chocolate Macaroons

Ingredients:
- Whole almonds (2 cups, about 9 1/2 ounces), toasted
- Sugar (1 cup)
- Ground cinnamon (1/4 teaspoon)
- Salt (1/8 teaspoon)
- Large egg (1)
- Large egg whites (2)
- Almond extract (1/4 teaspoon)
- Bittersweet or semisweet chocolate (3/4 cup, finely chopped)

Instructions: Position rack in center of oven and preheat to 350°F. Line 2 baking sheets with foil, butter the foil, and dust with flour. Finely grind almonds, sugar, cinnamon, and salt in a processor. Add egg, egg whites, and almond extract, processing until mixture holds together. Transfer to a bowl and stir in chocolate. Using moistened hands, roll mixture into 1-inch balls and place on prepared sheets, flattening to 1/3-inch thick rounds. Bake macaroons until tops puff and centers remain soft, about 12 minutes. Transfer to a rack to cool completely. Store in an airtight container at room temperature. Can be prepared up to 3 days ahead.

Example 5
Title: Scallop and Shrimp Creole 

Ingredients:
- 1 pound large shrimp in the shell, or approximately 3/4 pound frozen, cleaned shrimp, defrosted (about 30 pieces)
- 1 1/4 pounds sea or bay scallops
- 3 tablespoons bacon fat or vegetable oil
- 1/2 cup small diced or chopped onion
- 1/2 cup small diced or chopped green bell pepper
- 1/2 cup small diced or chopped red bell pepper
- 1/3 cup small diced or chopped celery
- 2 garlic cloves, finely chopped
- 3/4 teaspoon dried thyme leaves
- 1/4 teaspoon dried oregano leaves
- 1 bay leaf
- 1/4 teaspoon cayenne
- 2 teaspoons flour
- 1 cup crushed or chopped drained canned plum tomatoes, plus 1/4 cup of their juice
- Salt and freshly ground black or white pepper to taste
- Tabasco sauce to taste
- 1 tablespoon chopped fresh (flat-leaf or curly) parsley

Instructions: To prepare: Peel the shrimp and save the shells for shrimp oil or shrimp broth or discard them. Pick up a shrimp and make a shallow slit down the middle of the length of the back to expose the black intestine. Slit all the shrimp and lift out the black intestine with the point of your paring knife, or flush it out under cold running water. If using defrosted cleaned shrimp, skip this step. Either way, dry the shrimp well with paper towels and set them aside. Clean the sea scallops by peeling and discarding the little strip of muscle that is attached to one side. (If your scallops are somewhat old the muscle strip may not be there.) Place the scallops in a colander and wash them well under cold running water — keep an eye open for specks of dark sand. Drain the scallops well and roll them in paper towels to dry them thoroughly. If the scallops are very large, cut them into 1/2" to 3/4" pieces (it's the thickness that determines the cooking time, not how wide they are). If you're using bay scallops, don't remove the tiny strip of muscle, it's tender. Either way, set the scallops aside while you're preparing the sauce. Put the bacon fat or vegetable oil into a skillet or stew-type pot that's wide and deep enough to hold all the seafood in about 2 layers with about 2 cups of sauce. Place the skillet over medium-high heat and add the onion, peppers, and celery. Cook the vegetables, adjusting the heat if necessary and stirring frequently, until they have become slightly wilted and a little brown, about 5 minutes. When the vegetables are ready, turn the heat to low and stir in the garlic, thyme, oregano, bay leaf and cayenne, cooking for about 30 seconds. Add the flour and continue to stir for about 1 minute more to cook away its raw taste. Add the tomatoes and their juice, and simmer the sauce, covered, over very low heat, about 10 minutes, or until the vegetables are almost tender and it's very thick. Stir the sauce once or twice during this period. Season it generously with salt, pepper, and Tabasco, and set it aside. The sauce can be made 3 or 4 days in advance and refrigerated, but be sure to heat it through before continuing with the recipe. In a separate bowl, mix the scallops and shrimp together and season them well with salt and pepper. (If you're using bay scallops that are smaller than 1/2" in width, don't mix them with the shrimp at this point — see Note.) Remove about one half of the sauce to a bowl, spreading the remaining sauce evenly over the bottom of the skillet or pot. Distribute the seafood evenly over the sauce in the skillet and spread the reserved sauce over the top of the seafood — it won't completely cover. Place the skillet over medium-high heat and, without stirring, heat the mixture until you see 3 or 4 bubbles at the surface. Reduce the heat to very low, tightly cover the skillet, and gently simmer the mixture until the shrimp are white throughout and the scallops are slightly translucent in the center, 8 to 12 minutes, cutting one of each to check, if you're unsure. (If you prefer, you can bake the Creole in a preheated 325°F oven once the bubbles have come to the surface.) The Creole will be much thinner now from the shellfish juices. Taste it for seasoning and adjust it with salt, a generous amount of pepper, and additional Tabasco, if you like.
Peel the shrimp and save the shells for shrimp oil or shrimp broth or discard them. Pick up a shrimp and make a shallow slit down the middle of the length of the back to expose the black intestine. Slit all the shrimp and lift out the black intestine with the point of your paring knife, or flush it out under cold running water. If using defrosted cleaned shrimp, skip this step. Either way, dry the shrimp well with paper towels and set them aside.
Clean the sea scallops by peeling and discarding the little strip of muscle that is attached to one side. (If your scallops are somewhat old the muscle strip may not be there.) Place the scallops in a colander and wash them well under cold running water — keep an eye open for specks of dark sand. Drain the scallops well and roll them in paper towels to dry them thoroughly. If the scallops are very large, cut them into 1/2" to 3/4" pieces (it's the thickness that determines the cooking time, not how wide they are). If you're using bay scallops, don't remove the tiny strip of muscle, it's tender. Either way, set the scallops aside while you're preparing the sauce.
Put the bacon fat or vegetable oil into a skillet or stew-type pot that's wide and deep enough to hold all the seafood in about 2 layers with about 2 cups of sauce. Place the skillet over medium-high heat and add the onion, peppers, and celery. Cook the vegetables, adjusting the heat if necessary and stirring frequently, until they have become slightly wilted and a little brown, about 5 minutes. When the vegetables are ready, turn the heat to low and stir in the garlic, thyme, oregano, bay leaf and cayenne, cooking for about 30 seconds. Add the flour and continue to stir for about 1 minute more to cook away its raw taste. Add the tomatoes and their juice, and simmer the sauce, covered, over very low heat, about 10 minutes, or until the vegetables are almost tender and it's very thick. Stir the sauce once or twice during this period. Season it generously with salt, pepper, and Tabasco, and set it aside. The sauce can be made 3 or 4 days in advance and refrigerated, but be sure to heat it through before continuing with the recipe.
In a separate bowl, mix the scallops and shrimp together and season them well with salt and pepper. (If you're using bay scallops that are smaller than 1/2" in width, don't mix them with the shrimp at this point — see Note.) Remove about one half of the sauce to a bowl, spreading the remaining sauce evenly over the bottom of the skillet or pot. Distribute the seafood evenly over the sauce in the skillet and spread the reserved sauce over the top of the seafood — it won't completely cover. Place the skillet over medium-high heat and, without stirring, heat the mixture until you see 3 or 4 bubbles at the surface. Reduce the heat to very low, tightly cover the skillet, and gently simmer the mixture until the shrimp are white throughout and the scallops are slightly translucent in the center, 8 to 12 minutes, cutting one of each to check, if you're unsure. (If you prefer, you can bake the Creole in a preheated 325°F oven once the bubbles have come to the surface.)
The Creole will be much thinner now from the shellfish juices. Taste it for seasoning and adjust it with salt, a generous amount of pepper, and additional Tabasco, if you like.
To serve: Ladle the Creole into warm soup plates or bowls, discarding the bay leaf. Sprinkle each serving with chopped parsley and serve right away. <a name="note">Note:</a> If you've reserved bay scallops, add them approximately halfway through the cooking (depending on their size), pressing them into the simmering mixture as best as possible. Cover the skillet again and finish cooking.
Ladle the Creole into warm soup plates or bowls, discarding the bay leaf. Sprinkle each serving with chopped parsley and serve right away.
<a name="note">Note:</a> If you've reserved bay scallops, add them approximately halfway through the cooking (depending on their size), pressing them into the simmering mixture as best as possible. Cover the skillet again and finish cooking.

Example 5 cleaned
Title: Scallop and Shrimp Creole

Ingredients:
- Large shrimp in the shell, or approximately 3/4 pound frozen cleaned shrimp, defrosted (about 30 pieces)
- Sea or bay scallops (1 1/4 pounds)
- Bacon fat or vegetable oil (3 tablespoons)
- Onion, diced or chopped (1/2 cup)
- Green bell pepper, diced or chopped (1/2 cup)
- Red bell pepper, diced or chopped (1/2 cup)
- Celery, diced or chopped (1/3 cup)
- Garlic cloves, finely chopped (2)
- Dried thyme leaves (3/4 teaspoon)
- Dried oregano leaves (1/4 teaspoon)
- Bay leaf (1)
- Cayenne (1/4 teaspoon)
- Flour (2 teaspoons)
- Canned plum tomatoes, crushed or chopped, plus juice (1 cup tomatoes and 1/4 cup juice)
- Salt and freshly ground black or white pepper (to taste)
- Tabasco sauce (to taste)
- Fresh parsley, chopped (1 tablespoon)

Instructions: Peel shrimp and remove the intestine with a knife, flushing under cold water if necessary, then dry well with paper towels. Clean scallops, peeling and discarding the muscle strip, and pat dry; cut into 1/2" pieces if large. Heat bacon fat or oil in a wide skillet over medium-high heat. Add onion, peppers, and celery, cooking until lightly browned, about 5 minutes. Lower heat, stir in garlic, thyme, oregano, bay leaf, and cayenne, and cook 30 seconds. Add flour, stirring to remove raw taste, about 1 minute. Add tomatoes with juice and simmer, covered, on low for 10 minutes, or until thick. Season generously with salt, pepper, and Tabasco. Separately, season scallops and shrimp with salt and pepper. Remove half of sauce, spread the remaining evenly in skillet, and layer seafood over it. Cover with reserved sauce. Place skillet over medium-high heat until bubbling, then reduce heat to low and simmer until shrimp are opaque and scallops are slightly translucent, 8-12 minutes. Adjust seasoning and serve.

Now it is your turn. Here is a recipe in its un-cleaned form. Use the instructions above to clean this recipe and use the desired output format shown above. You must use hyphens to separate the components of the ingredient list. 

{recipe}
"""

FULL_CLEAN_TEMPLATE = PromptTemplate.from_template(full_clean_prompt)

#print(FULL_CLEAN_TEMPLATE.format(recipe='test'))