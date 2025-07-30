import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatMaritalk
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
# from prompts_without_preferences import LlamaPrompt, MaritacaPrompt, GPTPrompt
from prompts import LlamaPrompt, MaritacaPrompt, GPTPrompt
from prompts import BasePrompt

class SummaryBuilder():
    def __init__(self):
        self.output_parser = StrOutputParser()
        self.base_prompt = BasePrompt()
        load_dotenv()

    def get(self, person_name, news, txt_categoria, user_preferences, soccer_team, model="gpt"):
        self.person_name = person_name
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
    
    def get_prompt_to_persona(self):
        template = self.base_prompt.build_prompt_to_persona()

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", ""),  
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

        chain = template | llm
        prompt = chain.invoke({"person_name": self.person_name, "text_cat": self.cat, "user_prefs": self.user_preferences, "soccer_team": self.soccer_team}).content

        return prompt

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

        prompt_persona = self.get_prompt_to_persona()

        prompt = g_prompt.get(self.news, self.cat, self.user_preferences, self.soccer_team, prompt_persona)
        #prompt = g_prompt.get(self.news)

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", ""),  
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        chain = prompt | llm | self.output_parser

        response = chain.invoke({"input": {self.news}})
        print(response)
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