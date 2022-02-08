# kaggle_feedbackprize

코드를 실행하기 전, [kaggle 에서 전체 데이터](https://www.kaggle.com/c/feedback-prize-2021/data)만 다운로드 해주면 됩니다.

다운로드 후, repo 가 이하와 같은지 확인합니다.

    kaggle_feedbackprize
    ├─ input
    │    ├─ test
    │    ├─ train
    │    ├─ train.csv
    │    └─ train_5folds.csv
    │
    ├─ fb-sbert-yh
    │    ├─ README.md
    │    ├─ longformer-base-4096
    │    ├─ train_yh.py
    │    └─ utils_yh.py


확인 했다면, 이하 코드를 실행합니다.

    python train_yh.py --fold 0 --model longformer-base-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --small_batch --sbert
    

**cf**
* small_batch : 코드가 오류 없이 돌아가는지 확인하기 위한 parser. 전체 데이터에 대해 training 을 하고 싶다면 --no-small_batch 로 바꿔줍니다.
* sbert : sentence 정보를 넣어주는 parser. sbert 를 사용하고 싶지 않다면 --no-sbert 로 바꿔줍니다.  
* 다른 parser 의 의미는 코드를 확인 부탁드립니다.