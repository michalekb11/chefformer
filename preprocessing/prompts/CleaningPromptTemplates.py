from langchain_core.prompts import PromptTemplate

#=====Clean title======
title_clean_prompt = "You are an expert in cleaning recipe titles. Clean the following recipe title: {title}"

TITLE_CLEAN_TEMPLATE = PromptTemplate.from_template(title_clean_prompt)


#=====Clean instructions======
instruction_clean_prompt = "You are an expert in cleaning recipe titles. Clean the following recipe title: {title}"

INSTRUCTION_CLEAN_TEMPLATE = PromptTemplate.from_template(instruction_clean_prompt)