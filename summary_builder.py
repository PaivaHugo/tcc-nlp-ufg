import os
from langchain_community.chat_models import ChatMaritalk
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_ollama.llms import OllamaLLM
from prompts_without_preferences import LlamaPrompt, MaritacaPrompt, GPTPrompt

class SummaryBuilder():
    def __init__(self, news, model="llama"):
        self.news = news
        #self.user_preferences = user_preferences
        self.model = model
        self.output_parser = StrOutputParser()

    def get(self):
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
        # prompt = m_prompt.get(self.news, self.user_preferences)
        prompt = m_prompt.get(self.news)
        llm = ChatMaritalk(
            model=os.environ.get("MARITACA_MODEL", ""),  
            api_key=os.environ.get("MARITACA_API_KEY", ""),
            temperature=0.7,
            # max_tokens=1000,
        )
        chain = prompt | llm | self.output_parser

        response = chain.invoke({"input": {self.news}})
        return response
    
    def gpt(self):
        g_prompt = GPTPrompt()
        # prompt = g_prompt.get(self.news, self.user_preferences)
        prompt = g_prompt.get(self.news)
        llm = OpenAI(
            model=os.environ.get("OPENAI_MODEL", ""),  
            api_key=os.environ.get("OPENAI_API_KEY", ""),
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