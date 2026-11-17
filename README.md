# Triagem automática de tickets com LLM local

Classifica tickets de suporte (categoria, prioridade, sentimento) e gera um resumo curto usando um modelo rodando localmente via [Ollama](https://ollama.com). Nenhum dado de cliente sai da máquina.

## Como funciona

1. Lê um CSV com a coluna `mensagem`.
2. Para cada ticket, envia ao modelo um prompt com critérios de prioridade e exemplos (few-shot).
3. Valida a resposta com Pydantic: valores fora das listas permitidas são rejeitados (tolerando diferenças de acento e maiúsculas).
4. Se a saída for inválida, tenta novamente (até 3 vezes). Se continuar falhando, o ticket é marcado como **Revisão manual**, em vez de receber um valor padrão que mascararia o erro.
5. Salva o CSV final com as colunas `status`, `tentativas` e `erro` para auditoria.
6. `avaliar.py` compara o resultado com rótulos humanos (acurácia, precisão, recall e matriz de confusão).

## Estrutura do repositório

```
triagem.py            # pipeline de triagem
avaliar.py            # avaliação contra rótulos humanos
tickets_exemplo.csv   # 12 tickets fictícios com gabarito
requirements.txt
resultados/           # saídas de cada modelo (phi3.csv, llama3.csv)
```

## Pré-requisitos

- Python 3.9 ou superior
- [Ollama](https://ollama.com) instalado e em execução
- Dependências: `pip install -r requirements.txt`

## Executando

```bash
ollama pull phi3
pip install -r requirements.txt

python triagem.py --entrada tickets_exemplo.csv --saida tickets_triados.csv
python avaliar.py --arquivo tickets_triados.csv
```

Para testar outro modelo, baixe-o e passe o nome em `--modelo`:

```bash
ollama pull llama3
python triagem.py --entrada tickets_exemplo.csv --saida resultados/llama3.csv --modelo llama3
python avaliar.py --arquivo resultados/llama3.csv
```

## Exemplo de saída

| mensagem | categoria | prioridade | sentimento | resumo |
|---|---|---|---|---|
| Fui cobrado duas vezes na fatura deste mês. Quero o estorno imediatamente. | Financeiro | Urgente | Negativo | Cobrança duplicada, pede estorno imediato |

## Resultados (versão 1)

Avaliação em 12 tickets rotulados manualmente, com os modelos `phi3` e `llama3` rodando via Ollama. Os tempos de execução foram semelhantes nos dois.

| Campo | phi3 | llama3 |
|---|---|---|
| Categoria | 9/12 (75%) | 9/12 (75%) |
| Prioridade | 5/12 (42%) | 11/12 (92%) |
| Sentimento | 11/12 (92%) | 8/12 (67%) |
| Revisão manual | 0 | 0 |

As saídas completas estão em [`resultados/`](resultados/).

**Análise dos erros**
- phi3 tende a superestimar a prioridade: perguntas simples viram "Média" em vez de "Baixa".
- llama3 classifica perguntas neutras e educadas como "Positivo".
- Nenhum dos dois usou a categoria "Outros": elogios e sugestões foram para "Dúvida Geral". O prompt não definia essa categoria.
- A fronteira entre "Financeiro" e "Dúvida Geral" é ambígua: perguntas sobre nota fiscal e desconto foram acertadas por um modelo e erradas pelo outro.

**Limitações**
- Conjunto de avaliação pequeno (12 tickets): cada ticket vale ~8 pontos percentuais.
- Gabarito subjetivo, feito por uma única pessoa.
- Os exemplos few-shot do prompt são parecidos com alguns tickets de teste, o que pode ter inflado os acertos.

## Próximos passos

- Definir critérios explícitos de categoria (incluindo "Outros") e de sentimento no prompt, e trocar os exemplos few-shot por outros que não se pareçam com o conjunto de teste.
- Ampliar o gabarito para pelo menos 50 tickets e revisar os rótulos ambíguos.

