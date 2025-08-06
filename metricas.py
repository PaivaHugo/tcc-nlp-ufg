import json
import numpy as np

def metricas():
    
    # Carregue o JSON (você pode substituir por um arquivo ou colar diretamente aqui)
    with open('evaluation_prompt_few_shot.json', 'r') as f:
        data = json.load(f)

    maritaca_bert_f1_scores = []
    maritaca_rouge_f1_scores = []
    gpt_bert_f1_scores = []
    gpt_rouge_f1_scores = []

    person1_bert = []
    person2_bert = []
    person3_bert = []

    person1_rouge = []
    person2_rouge = []
    person3_rouge = []

    person1_bert_gpt = []
    person2_bert_gpt = []
    person3_bert_gpt = []

    person1_rouge_gpt = []
    person2_rouge_gpt = []
    person3_rouge_gpt = []


    # Percorre todos os textos e personas
    for item in data.values():
        personas = item["maritaca"]["personas"]
        for p in personas.values():
            maritaca_bert_f1_scores.append(p["bert_score"]["f1"])
            maritaca_rouge_f1_scores.append(p["rouge_l"]["f1"])



    # for i, item in data.items():
    #     personas = item["maritaca"]["personas"]
    #     for k, p in personas.items():
    #         if k == "1":
    #             person1_bert.append(p["bert_score"]["f1"])
    #             person1_rouge.append(p["rouge_l"]["f1"])
    #         elif k == "2":
    #             person2_bert.append(p["bert_score"]["f1"])
    #             person2_rouge.append(p["rouge_l"]["f1"])
    #         else:
    #             person3_bert.append(p["bert_score"]["f1"])
    #             person3_rouge.append(p["rouge_l"]["f1"])
            # maritaca_bert_f1_scores.append(p["bert_score"]["f1"])
            # maritaca_rouge_f1_scores.append(p["rouge_l"]["f1"])

    # for it, item in data.items():
    #     personas_ = item["gpt"]["personas"]
    #     for k_, p_ in personas_.items():
    #         if k_ == "1":
    #             person1_bert_gpt.append(p["bert_score"]["f1"])
    #             person1_rouge_gpt.append(p["rouge_l"]["f1"])
    #         elif k_ == "2":
    #             person2_bert_gpt.append(p["bert_score"]["f1"])
    #             person2_rouge_gpt.append(p["rouge_l"]["f1"])
    #         else:
    #             person3_bert_gpt.append(p["bert_score"]["f1"])
    #             person3_rouge_gpt.append(p["rouge_l"]["f1"])
    #         gpt_bert_f1_scores.append(p_["bert_score"]["f1"])
    #         gpt_rouge_f1_scores.append(p_["rouge_l"]["f1"])

    # person1_maritaca_bert = np.array(person1_bert)
    # print(f" person1_maritaca_bert: ")
    # print(f"  Média: {person1_maritaca_bert.mean():.4f}")
    # print(f"  Desvio padrão: {person1_maritaca_bert.std():.4f}")

    # person2_maritaca_bert = np.array(person2_bert)
    # print(f" person2_maritaca_bert: ")
    # print(f"  Média: {person2_maritaca_bert.mean():.4f}")
    # print(f"  Desvio padrão: {person2_maritaca_bert.std():.4f}")

    # person3_maritaca_bert = np.array(person3_bert)
    # print(f" person3_maritaca_bert: ")
    # print(f"  Média: {person3_maritaca_bert.mean():.4f}")
    # print(f"  Desvio padrão: {person3_maritaca_bert.std():.4f}") 

    # person1_gpt_bert = np.array(person1_bert_gpt)
    # print(f" person1_gpt_bert: ")
    # print(f"  Média: {person1_gpt_bert.mean():.4f}")
    # print(f"  Desvio padrão: {person1_gpt_bert.std():.4f}")

    # person2_gpt_bert = np.array(person2_bert_gpt)
    # print(f" person2_gpt_bert: ")
    # print(f"  Média: {person2_gpt_bert.mean():.4f}")
    # print(f"  Desvio padrão: {person2_gpt_bert.std():.4f}")

    # person3_gpt_bert = np.array(person3_bert_gpt)
    # print(f" person3_gpt_bert: ")
    # print(f"  Média: {person3_gpt_bert.mean():.4f}")
    # print(f"  Desvio padrão: {person3_gpt_bert.std():.4f}")

    # person1_maritaca_rouge = np.array(person1_rouge)
    # print(f" person1_maritaca_rouge: ")
    # print(f"  Média: {person1_maritaca_rouge.mean():.4f}")
    # print(f"  Desvio padrão: {person1_maritaca_rouge.std():.4f}")

    # person2_maritaca_rouge = np.array(person2_rouge)
    # print(f" person2_maritaca_rouge: ")
    # print(f"  Média: {person2_maritaca_rouge.mean():.4f}")
    # print(f"  Desvio padrão: {person2_maritaca_rouge.std():.4f}")

    # person3_maritaca_rouge = np.array(person3_rouge) 
    # print(f" person3_maritaca_rouge: ")
    # print(f"  Média: {person3_maritaca_rouge.mean():.4f}")
    # print(f"  Desvio padrão: {person3_maritaca_rouge.std():.4f}")

    # person1_gpt_rouge = np.array(person1_rouge_gpt)
    # print(f" person1_gpt_rouge: ")
    # print(f"  Média: {person1_gpt_rouge.mean():.4f}")
    # print(f"  Desvio padrão: {person1_gpt_rouge.std():.4f}")

    # person2_gpt_rouge = np.array(person2_rouge_gpt)
    # print(f" person2_gpt_rouge: ")
    # print(f"  Média: {person2_gpt_rouge.mean():.4f}")
    # print(f"  Desvio padrão: {person2_gpt_rouge.std():.4f}")

    # person3_gpt_rouge = np.array(person3_rouge_gpt)
    # print(f" person3_gpt_rouge: ")
    # print(f"  Média: {person3_gpt_rouge.mean():.4f}")
    # print(f"  Desvio padrão: {person3_gpt_rouge.std():.4f}")

    # Converte para array numpy para facilitar os cálculos
    maritaca_bert_f1 = np.array(maritaca_bert_f1_scores)
    maritaca_rouge_f1 = np.array(maritaca_rouge_f1_scores)

    # gpt_bert_f1 = np.array(gpt_bert_f1_scores)
    # gpt_rouge_f1 = np.array(gpt_rouge_f1_scores)

    # Cálculo das métricas
    print("BERTScore F1 Maritaca:")
    print(f"  Média: {maritaca_bert_f1.mean():.4f}")
    print(f"  Desvio padrão: {maritaca_bert_f1.std():.4f}")

    print("\nROUGE-L F1 Maritaca:")
    print(f"  Média: {maritaca_rouge_f1.mean():.4f}")
    print(f"  Desvio padrão: {maritaca_rouge_f1.std():.4f}")

    # # Cálculo das métricas
    # print("BERTScore F1 GPT:")
    # print(f"  Média: {gpt_bert_f1.mean():.4f}")
    # print(f"  Desvio padrão: {gpt_bert_f1.std():.4f}")

    # print("\nROUGE-L F1 GPT:")
    # print(f"  Média: {gpt_rouge_f1.mean():.4f}")
    # print(f"  Desvio padrão: {gpt_rouge_f1.std():.4f}")

def clear():
    with open("evaluation_duplo.json", "r") as e:
        evaluation = json.load(e)

    for k, v in evaluation.items():
        v["original_text"] = ""
        for y, x in v["maritaca"]["personas"].items():
            x["summary"] = ""

        for z, w in v["gpt"]["personas"].items():
            w["summary"] = ""

    with open("evaluation_duplo_alterado.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(evaluation, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    metricas()