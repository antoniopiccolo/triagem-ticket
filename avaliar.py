"""Avalia a qualidade da triagem comparando com rótulos feitos por humanos.

O CSV triado precisa ter, além das colunas geradas pelo modelo, as colunas de
gabarito: categoria_real, prioridade_real e sentimento_real.

Uso:
    python avaliar.py --arquivo tickets_triados.csv
"""
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CAMPOS = ["categoria", "prioridade", "sentimento"]


def main():
    parser = argparse.ArgumentParser(description="Avaliação da triagem")
    parser.add_argument("--arquivo", default="tickets_triados.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.arquivo)

    for campo in CAMPOS:
        real_col = f"{campo}_real"
        if real_col not in df.columns:
            print(f"[aviso] coluna '{real_col}' ausente, pulando {campo}.")
            continue

        base = df.dropna(subset=[real_col])
        y_real, y_pred = base[real_col], base[campo]

        print("=" * 60)
        print(f"{campo.upper()}  (n={len(base)})")
        print(f"Acurácia: {accuracy_score(y_real, y_pred):.2%}")
        print(classification_report(y_real, y_pred, zero_division=0))

        rotulos = sorted(set(y_real) | set(y_pred))
        matriz = pd.DataFrame(
            confusion_matrix(y_real, y_pred, labels=rotulos),
            index=[f"real: {r}" for r in rotulos],
            columns=[f"prev: {r}" for r in rotulos],
        )
        print("Matriz de confusão:")
        print(matriz, "\n")

    if "status" in df.columns:
        taxa = (df["status"] == "revisao_manual").mean()
        print(f"Taxa de tickets enviados para revisão manual: {taxa:.1%}")


if __name__ == "__main__":
    main()
