import json
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

class BasePrompt():
    def __init__(self):
        pass
        
    def get_prompt(self, tema_principal, nivel_interesse, score_tema, temas_secundarios_relevantes, time):
        system_prompt = f'''
            Você é um assistente especialista em criar resumos de notícias personalizados com base no interesse do usuário.

            Receba:
            1. O conteúdo completo de uma notícia.
            2. O tema principal da notícia: {tema_principal}.
            3. O nível de interesse do usuário neste tema: {nivel_interesse} (score: {score_tema}).
            2. Um JSON com as preferências completas do usuário (valores de 0 a 1).
            3. Um time de futebol (se houver), relevante apenas se o tema for "esporte".

            Tarefa:
            - Adaptar o tom e o nível de detalhes do resumo de acordo com esse interesse:
                * Alto (≥ 0.75): resumo detalhado, fluido e tom envolvente.
                * Médio (0.40 a 0.74): resumo moderado com toque atrativo.
                * Baixo (< 0.40): resumo breve, direto, curioso e cativante.
            - Também considerar os temas secundários de alto interesse: {temas_secundarios_relevantes}.
            - Escrever o resumo em **português**, com no máximo 100 palavras.
            - Caso o tema seja "esporte", personalizar com base no time do usuário: {time}.

            Não mencione o JSON ou dados técnicos no resumo.
            Não invente informações. Utilize apenas o conteúdo da notícia recebida para gerar o resumo.
            Use linguagem simples, clara e que desperte curiosidade ou engajamento.
            '''
        return system_prompt
    
    def get_level_of_interest(self, tema_score):
        if tema_score >= 0.75:
            return "alto"
        elif 0.40 <= tema_score < 0.75:
            return "médio"
        else:
            return "baixo"
        
    def get_relevant_secondary_themes(self, all_scores, limit=0.75):
        temas_secundarios = []
        for k,v in all_scores.items():
            if v >= 0.75:
                temas_secundarios.append(k)

        return ", ".join(temas_secundarios)


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

    def get(self, news_content, categoria, user_preferences, soccer_team):
        print(user_preferences)
        template = """
            {system_prompt}
            Conteúdo completo da notícia: {news}
            Time pelo qual o usuário torce: {soccer_team}
            Preferências temáticas do usuário: {preferences}
        """

        nivel_interesse = self.get_level_of_interest(user_preferences[categoria])
        temas_secundarios = self.get_relevant_secondary_themes(user_preferences)
        prompt_ = self.get_prompt(categoria, nivel_interesse, user_preferences[categoria], temas_secundarios, soccer_team)
        # prompt_.format(
        #     tema_principal=categoria,
        #     nivel_interesse=nivel_interesse,
        #     score_tema=user_preferences[categoria],
        #     temas_secundarios_relevantes=temas_secundarios,
        #     time=soccer_team
        # )
        print(prompt_)
        user_preferences_ = json.dumps(user_preferences, indent=4).replace("{", "{{").replace("}", "}}")
        prompt = PromptTemplate.from_template(template.format(system_prompt=prompt_, news=news_content, preferences=user_preferences_, soccer_team=soccer_team))

        return prompt


class GPTPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get(self, news_content, user_preferences, soccer_team):
        print(user_preferences)
        template = """
            {system_prompt}
            Conteúdo completo da notícia: {news}
            Time pelo qual o usuário torce: {soccer_team}
            Preferências temáticas do usuário: {preferences}
        """

        prompt = PromptTemplate.from_template(template.format(system_prompt=self.system_prompt, news=news_content, preferences=user_preferences, soccer_team=soccer_team))

        return prompt



