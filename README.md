Train a text to speech model in my own voice

project structure: 

TTS/
  ├── dataset/
    ├── audio/
    │   ├── 0001.wav
    │   ├── 0002.wav
    │   └── ...
    └── metadata.csv  (or .tsv / .jsonl depending on format)
  ├── main.py
  ├── README.md

metadata.csv examples:

0001|Hello, how are you?
0002|I'm training my voice for a TTS model.


dataset/
  ├── en/
  │   ├── wavs/
  │   │   └── 0001.wav
  │   └── metadata.csv     # 0001|This is an English sentence.
  ├── ur/
  │   ├── wavs/
  │   │   └── 0001.wav
  │   └── metadata.csv     # 0001|یہ اردو جملہ ہے۔

OR

combined_dataset/
  ├── wavs/
  │   ├── en_0001.wav
  │   ├── ur_0001.wav
  └── metadata.csv
        en_0001|<en>This is an English sentence.
        ur_0001|<ur>یہ اردو جملہ ہے۔

The decision is to train seperate models that know english and urdu or train one model that has additional language tokens it can train on



lambda labs ssh commadnd: 

ssh -i ~/.ssh/lambda-ssh-key ubuntu@165.1.71.6 