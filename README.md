# Triagem automática de tickets com LLM local

Classifica tickets de suporte (categoria, prioridade, sentimento) e gera um resumo curto usando um modelo rodando localmente via [Ollama](https://ollama.com). Nenhum dado de cliente sai da máquina.

## Como funciona

1. Lê um CSV com a coluna `mensagem`.
2. Para cada ticket, envia ao modelo um prompt com critérios de prioridade e exemplos (few-shot).
3. Valida a resposta com Pydantic: valores fora das listas permitidas são rejeitados (tolerando diferenças de acento e maiúsculas).
4. Se a saída for inválida, tenta novamente (até 3 vezes). Se continuar falhando, o ticket é marcado como **Revisão manual**, em vez de receber um valor padrão que mascararia o erro.
5. Salva o CSV final com as colunas `status`, `tentativas` e `erro` para auditoria.
6. `avaliar.py` compara o resultado com rótulos humanos (acurácia, precisão, recall e matriz de confusão).

## Executando

```bash
ollama pull phi3
pip install -r requirements.txt

python triagem.py --entrada tickets_exemplo.csv --saida tickets_triados.csv
python avaliar.py --arquivo tickets_triados.csv
```

Para testar outro modelo: `python triagem.py --modelo llama3.2`

## Comparando modelos

Rode o mesmo CSV com cada modelo, salve em arquivos diferentes e registre os resultados:

| Modelo | Acurácia categoria | Acurácia prioridade | Acurácia sentimento | % revisão manual |
|--------|--------------------|---------------------|---------------------|------------------|
| phi3   |                    |                     |                     |                  |
| llama3.2 |                  |                     |                     |                  |
| qwen2.5  |                  |                     |                     |                  |

## Próximos passos

- Ampliar o gabarito para 50 a 100 tickets rotulados à mão.
- Testar prompts diferentes (com e sem few-shot) e medir o ganho.
- Expor como API (FastAPI) e empacotar com Docker.
