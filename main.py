import csv

import pandas as pd

# import train


def extract_column():
    # phi_text_cleanの列を抽出する
    filename = "NTCIR10-17_phi_missing_augmented_rev_20260830"
    with open(f".data/{filename}.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 1行目のヘッダーをスキップ

        with open(f".data/{filename}.txt", "w", encoding="utf-8") as out:
            out.writelines(row[2] + "\n" for row in reader)


def run():
    # CSV読み込み
    df = pd.read_csv(".data/NTCIR10-17_train_v2.csv")

    # 2列目のテキスト一覧
    texts = df.iloc[:, 1].dropna().astype(str).tolist()
    dataset = [make_token_labels(text) for text in texts]

    # 確認
    for token, label in zip(dataset[745]["tokens"], dataset[745]["labels"]):
        print(token, label)


if __name__ == "__main__":
    extract_column()
    # run()
    # train.run()
