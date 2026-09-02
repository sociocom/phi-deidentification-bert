# 医療テキスト仮名化 Modern-BERT版

医療テキスト仮名化を行う[PHI-Deidentification](https://github.com/sociocom/PHI-Deidentification)のベースモデルをModern-BERTにしたものです。

NERの部分のみの実装で、LLMを使用した仮名化の実装は`PHI-Deidentification`を参照してください。

## Install

```
uv sync
```

## PHIタグ一覧

| タグ | 対象 |
|------|------|
| `<phi_age>` | 年齢 |
| `<phi_id>` | 識別番号 |
| `<phi_tel>` | 電話番号 |
| `<phi_job>` | 職業 |
| `<phi_location>` | 住所・地名 |
| `<phi_person>` | 人名 |
| `<phi_hospital>` | 医療機関名 |

## 学習

`configs/sample.yaml`をコピーしてconfigを作成します。

学習データは、１行１サンプルのタグ付きテキストです。
```
息子<phi_age>32歳</phi_age>生活習慣病の疑い
右眼の傷が悪化し、<phi_hospital>青山町中央病院</phi_hospital>眼科<phi_person>佐々木</phi_person>先生を受診した。
...
```

タグの種類は、configの`labels`で管理しますので、学習データに合わせて修正してください。

以下で実行します。
```
python train.py configs/sample.yaml
```

学習が完了したら、configの`train.output_dir`にモデルファイルが出力されます。


## 推論
テストデータは、１行１サンプルのplain-textです。
```
緊急連絡票には、避難先として香川県高松市青山町X-X-Xと記載されている。
佐々木ゆりさんから画像CDが届いた。
...
```

configの`inference.model_dir`に、学習されたモデルのディレクトリパスを指定します。
通常は、`train.output_dir`と同じです。

テストデータのパスは、`inference.data_path`に指定します。
推論結果は、`inference.output_path`にテキストファイルのパスを指定します。

以下で実行します。
```
python inference.py configs/sample.yaml
```
