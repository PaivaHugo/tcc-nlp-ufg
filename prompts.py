from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

class BasePrompt():
    def __init__(self):
        self.system_prompt = """"
        Você é um assistente especialista em criar resumos de notícias personalizados com base no interesse do usuário. 

        Você receberá:
        1. O conteúdo completo de uma notícia.
        2. Um JSON com as preferências temáticas do usuário (valores de 0 a 1 para cada tema, onde valores mais altos indicam maior interesse).

        Sua tarefa é:
        - Gerar um **resumo em português**, curto, claro e que **prenda a atenção** do usuário.
        - O tom deve ser **informativo e envolvente**, adaptado conforme o interesse do usuário no tema da notícia.
        - Se o score do tema principal for **maior ou igual a 0.75**, o resumo deve ser **mais encorpado**, com mais detalhes e nuance.
        - Se o score for **entre 0.40 e 0.74**, o resumo deve ser **moderado**, com as informações principais e um toque atrativo.
        - Se for **menor que 0.40**, produza um resumo **mais breve e direto**, ainda assim cativante o suficiente para despertar curiosidade.

        Importante:
        - Sempre escreva em **português**.
        - Use linguagem simples, mas com fluidez.
        - Nunca ultrapasse **100 palavras** no resumo.
        - Não inclua o JSON na resposta.

        Exemplo de JSON com preferências:
        {{
            "nome": "Camila",
            "id": 1,
            "scores": {{
                "tecnologia": 0.10,
                "economia": 0.50,
                "ciencia": 0.15,
                "saude": 0.89,
                "educacao": 0.93,
                "turismo": 0.90,
                "politica": 0.55,
                "esporte": 0.90
            }}
        }}

        """

class LlamaPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get(self, input_):
        template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """

        prompt = PromptTemplate.from_template(template.format(system_prompt=self.system_prompt, user_prompt=input_))

        return prompt


class MaritacaPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get(self, input_):
        template = """
            {system_prompt}
            {user_prompt}
        """

        prompt = PromptTemplate.from_template(template.format(system_prompt=self.system_prompt, user_prompt=input_))

        return prompt


class GPTPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get(self, input_):
        template = """
            {system_prompt}
            {user_prompt}
        """

        prompt = PromptTemplate.from_template(template.format(system_prompt=self.system_prompt, user_prompt=input_))

        return prompt



