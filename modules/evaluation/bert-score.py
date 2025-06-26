# Reference: https://github.com/haticeozbolat01/BERTScore-VS-ROUGE/tree/main
# Precisa exportar a variável de ambiente HF_TOKEN com o token do HuggingFace

from bert_score import BERTScorer
from rouge_score import rouge_scorer

class Evaluation():
    def __init__(self):
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def calculate_bert_score(self, full_text:str, summary:str) -> tuple:
        """Calcula o BERTScore entre um texto completo e seu resumo.

        Este método utiliza a biblioteca bert-score para computar a similaridade
        semântica entre um texto de referência (full_text) e um resumo candidato (summary).

        Args:
            full_text (str): O texto original de referência.
            summary (str): O resumo gerado para ser avaliado.

        Returns:
            tuple: Uma tupla contendo as métricas de precisão, recall e F1 do BERTScore como floats.
        """

        precision, recall, f1 = self.bert_scorer.score([full_text], [summary])
        
        precision = float(precision.mean())
        recall = float(recall.mean())
        f1 = float(f1.mean())
        
        return precision, recall, f1
    
    def calculate_rougeL(self, full_text:str, summary:str) -> tuple:
        """Calcula a métrica ROUGE-L entre um texto completo e seu resumo.

        Este método utiliza a biblioteca rouge-score para computar a sobreposição
        de subsequências mais longas (LCS) entre um texto de referência (full_text)
        e um resumo candidato (summary).

        Args:
            full_text (str): O texto original de referência.
            summary (str): O resumo gerado para ser avaliado.

        Returns:
            tuple: Uma tupla contendo as métricas de precisão, recall e F1 (fmeasure) do ROUGE-L como floats.
        """
        
        scores = self.rouge_scorer.score(full_text, summary)
        
        precision = scores['rougeL'].precision
        recall = scores['rougeL'].recall
        f1 = scores['rougeL'].fmeasure
        
        return precision, recall, f1
    


if __name__ == "__main__":
    evaluation = Evaluation()
    text1 = "Exemplo de texto completo para avaliação"
    text2 = "Exemplo de resumo para avaliação"
    bert = evaluation.calculate_bert_score(text1, text2)
    rouge_l = evaluation.calculate_rougeL(text1, text2)
    print(bert)
    print(rouge_l)