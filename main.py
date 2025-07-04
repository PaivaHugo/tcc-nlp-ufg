import json
from summary_builder import SummaryBuilder
from modules.evaluation import Evaluation

eval = Evaluation()

def get_all_summaries():
    summaries = {}
    model = "maritaca"
    summary = SummaryBuilder()

    with open("personas.json", "r") as p:
        personas = json.load(p)

    with open("materias.json", "r") as m:
        materias = json.load(m)

    
    for item in materias["materias"]:
        for persona in personas["personas"]:
            scores = json.dumps(persona["scores"], indent=4).replace("{", "{{").replace("}", "}}")
            text = summary.get(item["texto"], scores, persona["time_do_coracao"])

            if item["id"] in summaries:
                if persona["id"] in summaries[item["id"]][model]["personas"]:
                    summaries[item["id"]][model]["personas"][persona["id"]]["summary"] = text

                    score_bert_score = eval.calculate_bert_score(item["texto"], text)
                    summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["precision"] = score_bert_score[0]
                    summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["recall"] = score_bert_score[1]
                    summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["f1"] = score_bert_score[2]

                    score_rouge_l = eval.calculate_rougeL(item["texto"], text)
                    summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["precision"] = score_rouge_l[0]
                    summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["recall"] = score_rouge_l[1]
                    summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["f1"] = score_rouge_l[2]
                else:
                    summaries[item["id"]][model]["personas"][persona["id"]] = {
                                "summary": text,
                                "bert_score": {
                                "precision": 0,
                                "recall": 0,
                                "f1": 0
                            },
                            "rouge_l": {
                                "precision": 0,
                                "recall": 0,
                                "f1": 0
                            },
                            } 
                    score_bert_score = eval.calculate_bert_score(item["texto"], text)
                    summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["precision"] = score_bert_score[0]
                    summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["recall"] = score_bert_score[1]
                    summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["f1"] = score_bert_score[2]

                    score_rouge_l = eval.calculate_rougeL(item["texto"], text)
                    summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["precision"] = score_rouge_l[0]
                    summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["recall"] = score_rouge_l[1]
                    summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["f1"] = score_rouge_l[2]

            else:
                summaries[item["id"]] = {
                    "original_text": item["texto"],
                    model: {
                        "personas": {
                            persona["id"]: {
                                "summary": "",
                                "bert_score": {
                                "precision": 0,
                                "recall": 0,
                                "f1": 0
                            },
                            "rouge_l": {
                                "precision": 0,
                                "recall": 0,
                                "f1": 0
                            },
                            }    
                        }
                    
                    },
                }
                summaries[item["id"]][model]["personas"][persona["id"]]["summary"] = text

                score_bert_score = eval.calculate_bert_score(item["texto"], text)
                summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["precision"] = score_bert_score[0]
                summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["recall"] = score_bert_score[1]
                summaries[item["id"]][model]["personas"][persona["id"]]["bert_score"]["f1"] = score_bert_score[2]

                score_rouge_l = eval.calculate_rougeL(item["texto"], text)
                summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["precision"] = score_rouge_l[0]
                summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["recall"] = score_rouge_l[1]
                summaries[item["id"]][model]["personas"][persona["id"]]["rouge_l"]["f1"] = score_rouge_l[2]

    print(summaries)
    with open("evaluation.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(summaries, ensure_ascii=False, indent=4))
        


if __name__ == '__main__':
    news = "Sucesso de Marisa Maiô nas redes chama atenção para nova ferramenta de IA Uma nova ferramenta de inteligência artificial do Google está mudando a forma como vídeos são criados. Chamada de Veo 3, a tecnologia permite gerar cenas realistas com personagens, trilhas sonoras e até sotaques, tudo a partir de comandos de texto. Com a ferramenta, é possível criar vídeos com atores fictícios, expressões emocionais e falas personalizadas. Basta digitar um prompt (uma instrução à ferramenta) e o sistema transforma o texto em imagem e som. Veo 3: veja como funciona IA do Google“Ele consegue gerar vídeos realísticos de qualquer tipo de situação que você quiser. Isso apenas digitando comando de texto, o prompt de comando. É o roteiro que você vai dar para ela gerar aquele tipo de conteúdo”, explica o especialista em IA Brunno Sarttori.A Veo 3 também permite criar trilhas sonoras e efeitos com comandos simples, e controlar até o tom emocional dos personagens. “Você pode criar um vídeo de um ator e dizer a forma com que ele deve expressar aquelas palavras. Se ele deve estar triste, deve estar emocionado... É só vocês especificar no prompt de comando”, diz Sarttori. Arlindo Galvão, professor do Instituto de Informática da Universidade Federal de Goiás, afirma que “com pequenos comandos, feitos em linguagem natural, essa cotidiana, que a gente usa no dia a dia”, é possível conseguir resultados com uma qualidade impressionante. Um dos testes feitos pelo Fantástico usou o comando “Eu em Paris numa época marcante”. O resultado foi surpreendente: “O Fantástico veio até a Paris do século XIX, na construção da Torre Eiffel.” Em outra cena, um extraterrestre é entrevistado na praia do Rio de Janeiro. “Primeira vez no Rio, o que está achando da Terra?”, pergunta a repórter. O ET responde: “Cara, melhor que Júpiter, com certeza. Só achei o açaí meio caro.” A apresentadora fictícia Marisa Maiô também protagoniza momentos inusitados. Em uma cena, ela pergunta a uma médica: “Qual é o segredo para não ter mais espinhas?”. A resposta: “É só parar de ter.” Marisa conclui: “Obrigada, gente. Essa foi a inútil da médica.” O criador da personagem, Raony Philips, conta que muita gente acreditou que o programa era real. “Eu postei isso lá para a galera, tipo assim, uma grande zoeira, e do nada... tem gente falando: por que que a Marisa Maiô não é uma pessoa real, né?” Raony escreve os roteiros, cria as piadas e define as características dos personagens. Para ele, o elemento humano ainda é indispensável. “Sem aquele texto, sem a parte a parte humana por trás daquilo ali, jamais teria virado aquilo que virou. Eu estou muito surpreso ainda.” Do outro lado da equação, ferramentas de detecção de vídeos falsos estão sendo desenvolvidas para separar com clareza o que é conteúdo criado por IA e vídeos reais. Pesquisadores do Laboratório de Inteligência Artificial da Unicamp criaram uma tecnologia de detecção analisa rostos e outros elementos do vídeo em várias etapas. “Hoje, se você olha um vídeo ou imagem gerada por inteligência artificial, muito provavelmente você, como humano, não vai identificar que ela é falsa. Você precisa realmente de uma outra inteligência artificial para identificar”, diz Anderson Rocha, professor do Instituto de Computação da Unicamp. “Uma vez que o rosto é detectado, ele faz a análise. Os outros quadros são processos de análise do modelo”, diz Gabriel Bertocco, pesquisador da Unicamp. A ferramenta já está sendo usada em investigações do Ministério Público, mas ainda precisa ser atualizada para reconhecer os vídeos mais recentes gerados pela Veo 3. “É uma briga de um com o outro, uma brincadeira de gato e rato”, dizem os pesquisadores. As empresas responsáveis por essas ferramentas impõem limites éticos. Segundo Galvão, comandos que envolvem violência, abuso ou desinformação são bloqueados. “Ela tem umas linhas gerais, algumas diretrizes da própria empresa que, quando identificam algum comando que ultrapassa alguns limites, ela não gera. Não só abusos, mas também conteúdos que possam trazer desinformação.” Para artistas, o avanço da IA traz desafios e oportunidades. “Como artista, a gente morre de medo dessas coisas, né? Tipo assim, como que isso vai evoluir, que ponto que isso pode impactar no nosso trabalho”, diz Raony. “Mas ao mesmo tempo, eu acho que a gente tem que ver, né? Para entender: opa, é assim que funciona. Então ok, estou começando a entender.”A recomendação final é clara: desconfie. Questione o que vê."
    # user_preferences = '''{{
    #     "nome": "Camila",
    #     "id": 1,
    #     "scores": {{
    #         "tecnologia": 0.10,
    #         "economia": 0.50,
    #         "ciencia": 0.15,
    #         "saude": 0.89,
    #         "educacao": 0.93,
    #         "turismo": 0.90,
    #         "politica": 0.55,
    #         "esporte": 0.90
    #     }}
    # }}'''
    # summary = SummaryBuilder(news, model="maritaca")
    # summary = SummaryBuilder(news, user_preferences, model="maritaca")
    # text = summary.get()
    # print(text)

    get_all_summaries()