import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatMaritalk
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_ollama.llms import OllamaLLM
# from prompts_without_preferences import LlamaPrompt, MaritacaPrompt, GPTPrompt
from prompts import LlamaPrompt, MaritacaPrompt, GPTPrompt

class SummaryBuilder():
    def __init__(self):
        self.output_parser = StrOutputParser()
        load_dotenv()

    def get(self, news, txt_categoria, user_preferences, soccer_team, model="maritaca"):
        self.news = news
        self.cat = txt_categoria
        self.user_preferences = user_preferences
        self.soccer_team = soccer_team
        self.model = model
        
        models = {
            "maritaca": self.maritalk,
            "llama": self.llama,
            "gpt": self.gpt,
            "ollama": self.ollama
        }

        text = models[self.model]()
        return text

    def maritalk(self):
        m_prompt = MaritacaPrompt()
        prompt = m_prompt.get(self.news, self.cat, self.user_preferences, self.soccer_team)
        #prompt = m_prompt.get(self.news)
        llm = ChatMaritalk(
            model=os.getenv("MARITACA_MODEL", ""),  
            api_key=os.getenv("MARITACA_API_KEY", ""),
            temperature=0.7,
            # max_tokens=1000,
        )
        chain = prompt | llm | self.output_parser

        response = chain.invoke({"input": {self.news}})
        return response
    
    def gpt(self):
        g_prompt = GPTPrompt()
        prompt = g_prompt.get(self.news, self.user_preferences, self.soccer_team)
        #prompt = g_prompt.get(self.news)
        llm = OpenAI(
            model=os.getenv("OPENAI_MODEL", ""),  
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        chain = prompt | llm | self.output_parser

        response = chain.invoke({"input": {self.news}})
        return response

    def llama(self):
        l_prompt = LlamaPrompt()
        prompt = l_prompt.get(self.news)
        model = OllamaLLM(model="llama3.3")
        chain = prompt | model
        resp = chain.invoke({"input": self.news})
        print(resp)

    def ollama(self):
        model = OllamaLLM(model="deepseek-r1")
        chain = self.summary_prompt | model
        resp = chain.invoke({"input": self.news})
        print(resp)











  # should answer something like "1. Max\n2. Bella\n3. Charlie\n4. Rocky"