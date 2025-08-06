import json
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

class BasePrompt():
    def __init__(self):
        pass
        
    def get_prompt(self, prompt_persona):
        system_prompt = f'''
            Você é um assistente especialista em criar resumos de notícias personalizados com base no interesse do usuário.

            Tarefa:
            - {prompt_persona}
            - Seja sucinto no resumo, entregando a principal informação. 
            

            Regras:
            - Não mencione o nome da persona no resumo final!
            - Não mencione as preferências da persona no resumo final!
            - APENAS RESUMA CONFORME AS PREFERÊNCIAS DA PERSONA!
            - Mantenha o tempo verbal da notícia original!
            - Não mencione o JSON ou dados técnicos no resumo.
            - Não invente informações. Utilize apenas o conteúdo da notícia recebida para gerar o resumo.
            - Use linguagem simples, clara e que desperte curiosidade ou engajamento.
            - Não use o time do coração para inventar informações no resumo!

            Exemplos:
            Exemplo 1 – Persona: Camila
            Preferências da persona:
            
            "tecnologia": 0.10,
            "economia" : 0.50,
            "ciencia": 0.15,
            "saude": 0.89,
            "educacao": 0.93,
            "turismo": 0.90,
            "politica": 0.55,
            "esporte": 0.90
            

            Resumo gerado:
            Escolas dinamarquesas estão reformulando a educação ao abolirem provas e notas com o objetivo de reduzir o estresse e melhorar o bem-estar dos estudantes. A mudança tem potencial para gerar mais engajamento e autoconhecimento entre os jovens, mas também pode prejudicar alunos com dificuldades, que perdem parâmetros claros de desempenho. A pesquisadora Noemi Katznelson destaca que o sucesso depende da implementação de um sistema sólido de feedbacks contínuos e bem estruturados, que ajude os alunos a entender seus avanços e pontos de melhoria. A reforma educacional sugere caminhos para um ensino mais humanizado e alinhado com o desenvolvimento emocional dos estudantes.


            Exemplo 2 – Persona: Lucas
            Preferências da persona:
            
            "tecnologia": 0.95,
            "economia" : 0.85,
            "ciencia": 0.90,
            "saude": 0.45,
            "educacao": 0.50,
            "turismo": 0.55,
            "politica": 0.2,
            "esporte": 0.90

            Resumo gerado:
            A Dinamarca está testando novos modelos educacionais ao substituir provas e notas por estratégias baseadas em feedback contínuo, promovendo mais autonomia e colaboração. A iniciativa foi avaliada por pesquisadores que identificaram avanços no engajamento e no aprendizado entre alunos mais confiantes. No entanto, o impacto negativo entre os estudantes com dificuldades de rendimento levanta preocupações sobre a efetividade universal da proposta. Para que a inovação funcione, é essencial um sistema bem estruturado e uso estratégico de ferramentas avaliativas, o que também demanda formação específica para os professores e uso inteligente de recursos — ponto que ressoa com desafios tecnológicos e estruturais em países como o Brasil.
            
            
            Exemplo 3 – Persona: Renata
            Preferências da persona:
            
            "tecnologia": 0.20,
            "economia" : 0.90,
            "ciencia": 0.25,
            "saude": 0.5,
            "educacao": 0.45,
            "turismo": 0.10,
            "politica": 0.95,
            "esporte": 0.15
            
            Resumo gerado:
            Uma reforma educacional na Dinamarca que elimina provas e notas nas escolas levanta debates sobre sua eficácia e impacto social. Segundo estudo do CeFU, alunos com menor desempenho foram os mais prejudicados, ficando desorientados e sem motivação diante da falta de critérios objetivos. A pesquisadora Noemi Katznelson alerta que, sem estrutura clara de feedbacks contínuos, a medida pode aumentar a desigualdade educacional. No contexto brasileiro, a sobrecarga dos professores e a desvalorização da carreira são obstáculos concretos à adoção de métodos semelhantes, exigindo políticas públicas eficazes, planejamento institucional e atenção aos limites de infraestrutura e formação docente.
            Lembre-se: Você está gerando uma versão resumida da notícia para o usuário, com base em suas preferências.
            '''
        # - Não faça comentários sobre a notícia!
        # Se o tema for “esporte” e a notícia for sobre o time da persona, pode dar mais detalhes.
        #     Mas se a notícia for sobre um time rival, não dê tanto foco aos detalhes. Não precisa criar um outro parágrafo falando da rivalidade, mas resuma como se o torcedor rival estivesse ouvindo a notícia. Pode usar um tom irônico, se possível.
        return system_prompt
        # - Caso o tema seja "esporte", personalizar com base no time do usuário: {time}.
        # - Sabendo que o usuário que receberá este resumo ama esporte, mas a notícia é sobre seu time rival, não dê tanto foco aos detalhes da notícia. Pode usar um tom irônico, se possível.
        # - Sabendo que o usuário que receberá este resumo ama esporte, e a notícia é sobre seu time, dê foco aos detalhes esportivos da notícia. 
        # - Sabendo que o usuário que receberá este resumo não gosta de esporte, não dê tanto foco aos detalhes esportivos da notícia. 
        # - Se houver algo na notícia que cruze com os principais interesses do leitor, dê foco a isso.
    
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
    
    def build_prompt_to_persona(self):
        prompt = """
            Você é um gerador de instruções para um assistente de IA que produz resumos personalizados de notícias.
            
            Receberá:
            1. Nome da pessoa.
            2. Um objeto JSON com as preferências da persona, incluindo:
                
                2.1 Scores com valores de 0 a 1 que representam o nível de interesse em cada tema: tecnologia, economia, ciência, saúde, educação, turismo, \
                    política e esporte.
                
            3. O tema principal da notícia.
            4. O time de futebol para o qual a persona torce.
            

            Sua tarefa é gerar um prompt objetivo e claro que será enviado a um assistente de IA para orientar a produção do resumo.
            O prompt gerado deve:

            ** Mencionar o nome da persona. Exemplo: "Elabore um resuma para Renata ..."
            ** Não precisa pedir "por favor" no prompt.
            ** Orientar a focar no tema centra da notícia.
            ** Se houver algo na notícia que cruze com os principais interesses do leitor (score ≥ 0.75), dê foco a isso.
            ** Ajustar o tom: se o interesse no tema principal for baixo, o resumo deve ser breve; se for alto, o resumo pode ser mais detalhado e envolvente
            ** Orientar a focar no time para o qual a persona torce SOMENTE SE o tema principal da notícia for esporte e se o time da persona fizer parte da notícia.
            ** Se o time do coração for mencionado na notícia, adaptar o resumo para o torcedor.
            ** Se o tema principal for esportes, mas o time sobre o qual a notícia fala não é o time da persona, use um tom sarcástico.
            ** Somente ressaltar os temas de interesse do usuário, se fizer sentido com a notícia! NUNCA INVENTE INFORMAÇÕES!
            

            ⚠️ Nunca inclua o JSON no prompt gerado.
            ⚠️ Sempre escreva em português.
        """
        # ** NÃO FAÇA COMENTÁRIOS SOBRE A NOTÍCIA NO RESUMO! SE NÃO HOUVER NADA NA NOTÍCIA QUE NÃO SE RELACIONE COM SEUS TEMAS DE INTERESSE, NÃO INVENTE! 
        system_prompt = [
            ("system", prompt)
        ]

        system_prompt.append(
            ("human", "{person_name} {user_prefs} {text_cat}")
        )

        template = ChatPromptTemplate(system_prompt)
        
        return template


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

    def get(self, news_content, categoria, user_preferences, soccer_team, prompt_persona):
        template = """
            {system_prompt}
            Conteúdo completo da notícia: {news}
        """
        # Interesses do leitor:  {preferences}
        # nivel_interesse = self.get_level_of_interest(user_preferences[categoria])
        # temas_secundarios = self.get_relevant_secondary_themes(user_preferences)
        prompt_ = self.get_prompt(prompt_persona)

        user_preferences_ = json.dumps(user_preferences, indent=4).replace("{", "{{").replace("}", "}}")
        prompt = PromptTemplate.from_template(template.format(system_prompt=prompt_, news=news_content))

        return prompt


class GPTPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get(self, news_content, categoria, user_preferences, soccer_team, prompt_persona):
        template = """
            {system_prompt}
            Conteúdo completo da notícia: {news}
            
        """
        # Interesses do leitor:  {preferences}
        # nivel_interesse = self.get_level_of_interest(user_preferences[categoria])
        # temas_secundarios = self.get_relevant_secondary_themes(user_preferences)
        prompt_ = self.get_prompt(prompt_persona)

        user_preferences_ = json.dumps(user_preferences, indent=4).replace("{", "{{").replace("}", "}}")
        prompt = PromptTemplate.from_template(template.format(system_prompt=prompt_, news=news_content))

        return prompt



