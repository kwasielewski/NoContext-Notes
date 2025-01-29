import json
import os
from datetime import datetime
from subprocess import DEVNULL, run, PIPE

with open("train.json") as tf:
    songs = json.load(tf)

count = 0
wrong_songs = []
len_ = 1000

catalog_name = f'run{datetime.now().strftime("%Y-%m-%d%H-%M-%S")}'
os.mkdir(catalog_name)
for idx, song in enumerate(songs[:len_]):
    p = run(["DatasetBuilder/_build/default/bin/main.exe"], stdout=PIPE, stderr=DEVNULL, input=song["abc notation"], encoding='ascii')
    count += 1 if p.returncode != 0 else 0

    if p.returncode == 0:
        with open(f"{catalog_name}/{idx}.abc", "w") as f:
            f.write(p.stdout)
    else:
        wrong_songs += [song]
    
    if idx % 100 == 0:
        print(100.0*idx/len_,"%", sep='')

print(f"Res: {len_ - count}/{len_}")
print(f"Pred res: {(len_ - count) * (len(songs) / len_)}")

with open(f'{catalog_name}/bad.json', 'w') as f:
    json.dump(wrong_songs, f)
