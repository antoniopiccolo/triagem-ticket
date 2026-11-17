"""Triagem automática de tickets de suporte com um LLM local (Ollama).

Uso:
    python triagem.py --entrada tickets_exemplo.csv --saida tickets_triados.csv
"""
import argparse
import json
import sys
import unicodedata

import ollama
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

CATEGORIAS = ["Financeiro", "Suporte Técnico", "Dúvida Geral", "Outros"]
PRIORIDADES = ["Baixa", "Média", "Alta", "Urgente"]
SENTIMENTOS = ["Positivo", "Neutro", "Negativo"]
MAX_PALAVRAS_RESUMO = 10
TENTATIVAS = 3
REVISAO_MANUAL = "Revisão manual"

SYSTEM_PROMPT = f"""Você é um assistente de triagem de tickets de suporte.
Para cada mensagem de cliente, retorne APENAS um objeto JSON válido com exatamente estas chaves:
- "categoria": uma entre {CATEGORIAS}
- "prioridade": uma entre {PRIORIDADES}
- "sentimento": uma entre {SENTIMENTOS}
- "resumo": resumo do problema com no máximo {MAX_PALAVRAS_RESUMO} palavras.

Critérios de prioridade:
- Urgente: serviço totalmente parado, cobrança indevida grave, risco de perda financeira ou de dados.
- Alta: problema que impede o uso normal, mas existe alternativa temporária.
- Média: problema incômodo que não impede o uso.
- Baixa: dúvidas simples, elogios e sugestões."""

# Exemplos (few-shot): ajudam modelos pequenos a manter o formato e os critérios.
EXEMPLOS = [
    (
        "Fui cobrado duas vezes no cartão este mês e preciso do estorno hoje!",
        {
            "categoria": "Financeiro",
            "prioridade": "Urgente",
            "sentimento": "Negativo",
            "resumo": "Cobrança duplicada no cartão, pede estorno",
        },
    ),
    (
        "O aplicativo fecha sozinho quando abro a tela de relatórios.",
        {
            "categoria": "Suporte Técnico",
            "prioridade": "Alta",
            "sentimento": "Negativo",
            "resumo": "App fecha ao abrir tela de relatórios",
        },
    ),
    (
        "Vocês têm plano para equipes? Adorei o produto, só queria saber os valores.",
        {
            "categoria": "Dúvida Geral",
            "prioridade": "Baixa",
            "sentimento": "Positivo",
            "resumo": "Pergunta sobre valores do plano para equipes",
        },
    ),
]


def _normalizar(texto: str) -> str:
    """Remove acentos e caixa para comparar valores de forma tolerante."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.strip().lower()


def _casar(valor, opcoes: list[str]) -> str:
    """Devolve o valor canônico da lista ou levanta erro se não existir."""
    mapa = {_normalizar(o): o for o in opcoes}
    chave = _normalizar(str(valor))
    if chave not in mapa:
        raise ValueError(f"valor '{valor}' fora de {opcoes}")
    return mapa[chave]


class Triagem(BaseModel):
    """Esquema validado da resposta do modelo."""

    categoria: str
    prioridade: str
    sentimento: str
    resumo: str

    @field_validator("categoria")
    @classmethod
    def valida_categoria(cls, v):
        return _casar(v, CATEGORIAS)

    @field_validator("prioridade")
    @classmethod
    def valida_prioridade(cls, v):
        return _casar(v, PRIORIDADES)

    @field_validator("sentimento")
    @classmethod
    def valida_sentimento(cls, v):
        return _casar(v, SENTIMENTOS)

    @field_validator("resumo")
    @classmethod
    def corta_resumo(cls, v):
        return " ".join(v.split()[:MAX_PALAVRAS_RESUMO])


def montar_mensagens(msg: str) -> list[dict]:
    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
    for texto, saida in EXEMPLOS:
        mensagens.append({"role": "user", "content": f'Mensagem do cliente: "{texto}"'})
        mensagens.append(
            {"role": "assistant", "content": json.dumps(saida, ensure_ascii=False)}
        )
    mensagens.append({"role": "user", "content": f'Mensagem do cliente: "{msg}"'})
    return mensagens


def classificar(modelo: str, msg: str) -> dict:
    """Classifica um ticket. Tenta até TENTATIVAS vezes antes de pedir revisão manual."""
    ultimo_erro = ""
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            resposta = ollama.chat(
                model=modelo,
                messages=montar_mensagens(msg),
                format="json",
                options={"temperature": 0},
            )
            conteudo = resposta["message"]["content"].strip()
            dados = Triagem.model_validate_json(conteudo).model_dump()
            return {**dados, "status": "ok", "tentativas": tentativa, "erro": ""}
        except (ValidationError, json.JSONDecodeError) as e:
            ultimo_erro = f"saída inválida: {str(e)[:120]}"
        except Exception as e:  # falha de conexão, modelo ausente etc.
            ultimo_erro = f"{type(e).__name__}: {str(e)[:120]}"

    return {
        "categoria": REVISAO_MANUAL,
        "prioridade": REVISAO_MANUAL,
        "sentimento": REVISAO_MANUAL,
        "resumo": "",
        "status": "revisao_manual",
        "tentativas": TENTATIVAS,
        "erro": ultimo_erro,
    }


def main():
    parser = argparse.ArgumentParser(description="Triagem de tickets com LLM local")
    parser.add_argument("--entrada", default="tickets_brutos.csv")
    parser.add_argument("--saida", default="tickets_triados.csv")
    parser.add_argument("--modelo", default="phi3")
    args = parser.parse_args()

    print(f"Carregando tickets de {args.entrada}...")
    df = pd.read_csv(args.entrada)
    if "mensagem" not in df.columns:
        sys.exit("Erro: o CSV precisa ter uma coluna chamada 'mensagem'.")
    df = df.reset_index(drop=True)  # garante alinhamento posição a posição

    resultados = []
    for i, msg in enumerate(df["mensagem"].fillna("").astype(str), start=1):
        print(f"[{i}/{len(df)}] {msg[:50]}...")
        if not msg.strip():
            resultados.append(
                {
                    "categoria": REVISAO_MANUAL,
                    "prioridade": REVISAO_MANUAL,
                    "sentimento": REVISAO_MANUAL,
                    "resumo": "",
                    "status": "revisao_manual",
                    "tentativas": 0,
                    "erro": "mensagem vazia",
                }
            )
            continue
        resultados.append(classificar(args.modelo, msg))

    df_final = pd.concat([df, pd.DataFrame(resultados)], axis=1)
    df_final.to_csv(args.saida, index=False, encoding="utf-8-sig")

    total_manual = (df_final["status"] == "revisao_manual").sum()
    print(f"\nFinalizado. Arquivo gerado: {args.saida}")
    print(f"Tickets para revisão manual: {total_manual} de {len(df_final)}")
    print("\n--- Resumo por categoria ---")
    print(df_final["categoria"].value_counts())
    print("\n--- Resumo por prioridade ---")
    print(df_final["prioridade"].value_counts())


if __name__ == "__main__":
    main()
