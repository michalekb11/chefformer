from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from os import getenv

load_dotenv()

class GeminiFlash():
    def __init__(self, prompt_template: str = None, temperature: float = 0.3):
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.llm = self.load_llm()
        self.chain = self.load_chain()

    def load_llm(self):
        return GoogleGenerativeAI(model='models/gemini-1.5-flash-001',
                                  google_api_key=getenv('GOOGLE_API_KEY'),
                                  temperature=self.temperature)
    
    def load_chain(self):
        return self.prompt_template | self.llm
    
    def invoke(self, input_vars):
        return self.chain.invoke(input_vars)
    

