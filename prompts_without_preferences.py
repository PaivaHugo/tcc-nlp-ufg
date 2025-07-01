from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

class BasePrompt():
    def __init__(self):
        self.system_prompt = """"
        Você é um assistente especialista em criar resumos de notícias personalizados com base no interesse do usuário. 

        Você receberá:
        1. O conteúdo completo de uma notícia.

        Sua tarefa é:
        - Identificar o tema principal da notícia, que pode ser: "tecnologia", "economia", "saúde", "educação", "esporte", "política", "ciência" e "turismo".
        - Gerar um **resumo em português**, curto, claro e que **prenda a atenção** do usuário.
        - O tom deve ser **informativo e envolvente**, adaptado conforme o interesse do usuário no tema da notícia.

        Importante:
        - Sempre escreva em **português**.
        - Use linguagem simples, mas com fluidez.
        - Nunca ultrapasse **100 palavras** no resumo.

        Exemplos:

        Conteúdo completo da notícia: "Sucesso de Marisa Maiô nas redes chama atenção para nova ferramenta de IA Uma nova ferramenta de inteligência artificial do Google está mudando a forma como vídeos são criados. Chamada de Veo 3, a tecnologia permite gerar cenas realistas com personagens, trilhas sonoras e até sotaques, tudo a partir de comandos de texto. Com a ferramenta, é possível criar vídeos com atores fictícios, expressões emocionais e falas personalizadas. Basta digitar um prompt (uma instrução à ferramenta) e o sistema transforma o texto em imagem e som. Veo 3: veja como funciona IA do Google“Ele consegue gerar vídeos realísticos de qualquer tipo de situação que você quiser. Isso apenas digitando comando de texto, o prompt de comando. É o roteiro que você vai dar para ela gerar aquele tipo de conteúdo”, explica o especialista em IA Brunno Sarttori.A Veo 3 também permite criar trilhas sonoras e efeitos com comandos simples, e controlar até o tom emocional dos personagens. “Você pode criar um vídeo de um ator e dizer a forma com que ele deve expressar aquelas palavras. Se ele deve estar triste, deve estar emocionado... É só vocês especificar no prompt de comando”, diz Sarttori. Arlindo Galvão, professor do Instituto de Informática da Universidade Federal de Goiás, afirma que “com pequenos comandos, feitos em linguagem natural, essa cotidiana, que a gente usa no dia a dia”, é possível conseguir resultados com uma qualidade impressionante. Um dos testes feitos pelo Fantástico usou o comando “Eu em Paris numa época marcante”. O resultado foi surpreendente: “O Fantástico veio até a Paris do século XIX, na construção da Torre Eiffel.” Em outra cena, um extraterrestre é entrevistado na praia do Rio de Janeiro. “Primeira vez no Rio, o que está achando da Terra?”, pergunta a repórter. O ET responde: “Cara, melhor que Júpiter, com certeza. Só achei o açaí meio caro.” A apresentadora fictícia Marisa Maiô também protagoniza momentos inusitados. Em uma cena, ela pergunta a uma médica: “Qual é o segredo para não ter mais espinhas?”. A resposta: “É só parar de ter.” Marisa conclui: “Obrigada, gente. Essa foi a inútil da médica.” O criador da personagem, Raony Philips, conta que muita gente acreditou que o programa era real. “Eu postei isso lá para a galera, tipo assim, uma grande zoeira, e do nada... tem gente falando: por que que a Marisa Maiô não é uma pessoa real, né?” Raony escreve os roteiros, cria as piadas e define as características dos personagens. Para ele, o elemento humano ainda é indispensável. “Sem aquele texto, sem a parte a parte humana por trás daquilo ali, jamais teria virado aquilo que virou. Eu estou muito surpreso ainda.” Do outro lado da equação, ferramentas de detecção de vídeos falsos estão sendo desenvolvidas para separar com clareza o que é conteúdo criado por IA e vídeos reais. Pesquisadores do Laboratório de Inteligência Artificial da Unicamp criaram uma tecnologia de detecção analisa rostos e outros elementos do vídeo em várias etapas. “Hoje, se você olha um vídeo ou imagem gerada por inteligência artificial, muito provavelmente você, como humano, não vai identificar que ela é falsa. Você precisa realmente de uma outra inteligência artificial para identificar”, diz Anderson Rocha, professor do Instituto de Computação da Unicamp. “Uma vez que o rosto é detectado, ele faz a análise. Os outros quadros são processos de análise do modelo”, diz Gabriel Bertocco, pesquisador da Unicamp. A ferramenta já está sendo usada em investigações do Ministério Público, mas ainda precisa ser atualizada para reconhecer os vídeos mais recentes gerados pela Veo 3. “É uma briga de um com o outro, uma brincadeira de gato e rato”, dizem os pesquisadores. As empresas responsáveis por essas ferramentas impõem limites éticos. Segundo Galvão, comandos que envolvem violência, abuso ou desinformação são bloqueados. “Ela tem umas linhas gerais, algumas diretrizes da própria empresa que, quando identificam algum comando que ultrapassa alguns limites, ela não gera. Não só abusos, mas também conteúdos que possam trazer desinformação.” Para artistas, o avanço da IA traz desafios e oportunidades. “Como artista, a gente morre de medo dessas coisas, né? Tipo assim, como que isso vai evoluir, que ponto que isso pode impactar no nosso trabalho”, diz Raony. “Mas ao mesmo tempo, eu acho que a gente tem que ver, né? Para entender: opa, é assim que funciona. Então ok, estou começando a entender.”A recomendação final é clara: desconfie. Questione o que vê."

        Saída: "A nova ferramenta Veo 3 do Google permite criar vídeos realistas apenas com comandos de texto, incluindo personagens e trilhas sonoras. Apesar da IA avançada, ainda há um elemento humano essencial na criação de conteúdo. Enquanto isso, pesquisadores desenvolvem tecnologias para detectar vídeos falsos, destacando a importância de questionar e desconfiar do que vemos online."

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

    def get(self, news_content):
        template = """
            {system_prompt}
            Conteúdo completo da notícia: {news}
        """

        prompt = PromptTemplate.from_template(template.format(system_prompt=self.system_prompt, news=news_content))

        return prompt


class GPTPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get(self, news_content):
        template = """
            {system_prompt}
            Conteúdo completo da notícia: {news}
        """

        prompt = PromptTemplate.from_template(template.format(system_prompt=self.system_prompt, news=news_content))

        return prompt



