import numpy as np

# 依存関係の行列 A (問題の依存関係)
A = np.array([
    [0, 0, 0, 0, 0],  # 問題1は他の問題に依存しない
    [1, 0, 0, 0, 0],  # 問題2は問題1に依存
    [1, 1, 0, 0, 0],  # 問題3は問題1と問題2に依存
    [0, 1, 0, 0, 0],  # 問題4は問題2に依存
    [1, 1, 1, 0, 0]   # 問題5は問題1、問題2、問題3に依存
])


# 次に正解する確率を求める関数
def calculate_transition_probabilities(A, X):
    n = len(X)
    raw_probabilities = np.zeros(n)  # 遷移確率の元となる値
    
    # 不正解の問題に対して遷移確率を計算
    for i in range(n):
        if X[i] == 0:  # 不正解の問題のみ計算
            required_problems = A[i, :]  # i番目の問題に必要な依存関係
            
            # 依存関係がすべて0であるかを確認
            if np.all(required_problems == 0):
                raw_probabilities[i] =  10 # 必要なら、この問題の遷移確率を設定
                continue
            
            solved_problems = X * required_problems  # 実際に解けた問題
            
            num_required = np.sum(required_problems)  # 必要な問題の数
            num_solved = np.sum(solved_problems)      # 実際に解けた問題の数
            
            if num_required > 0:
                raw_probabilities[i] = np.exp(num_solved / num_required)
    
    # 総和で割って正規化
    total_sum = np.sum(raw_probabilities)  # expの総和
    if total_sum > 0:  # 総和が0でなければ正規化
        probabilities = raw_probabilities / total_sum
    else:
        probabilities = raw_probabilities  # 総和が0ならそのまま
    
    return probabilities


# 生徒の回答状況 X (1が正解、0が不正解)
# 例: 問題1と問題2は正解、問題3と問題4と問題5は不正解
X = np.array([1, 1, 0, 0, 0])

# 遷移確率の計算
probabilities = calculate_transition_probabilities(A, X)

# データセットを生成する関数
def generate_dataset(A, initial_X, num_correct_problems, num_data_per_step):
    n = len(initial_X)  # 問題数（今回は5）
    dataset = []
    
    # 正解させたい問題数ごとにデータを生成
    for i in range(num_correct_problems + 1):  # 0から正解問題数 i
        for j in range(num_data_per_step):  # 各ステップごとにデータ数
            X = initial_X.copy()  # 初期状態からスタート
            for k in range(i):  # i問正解させる
                probabilities = calculate_transition_probabilities(A, X)
                
                if np.sum(probabilities) > 0:  # 正規化された確率がある場合
                    # 確率に基づいて次に正解させる問題を選択
                    next_correct_problem = np.random.choice(n, p=probabilities)
                    X[next_correct_problem] = 1  # 選ばれた問題を正解に遷移させる
            
            dataset.append(X.copy())  # データセットに追加
    
    return np.array(dataset)

# データセット生成
num_correct_problems = 5  # 0問から5問まで正解
num_data_per_step = 100     # 各ステップごとに生成するデータ数

# 生徒の回答状況 X (1が正解、0が不正解)
# 初期状態は全て不正解
X_init = np.array([0, 0, 0, 0, 0])

dataset = generate_dataset(A, X_init, num_correct_problems, num_data_per_step)

# 結果を表示
print("生成されたデータセット:")
print(dataset)

import itertools
from collections import defaultdict

num_questions = 5

# 問題数5問に対する全ての可能な状態 (2^5 = 32状態)
states = list(itertools.product([0, 1], repeat=num_questions))
print(f"Number of States: {len(states)}\n")

# 各状態（ノード）に生徒を配置する (状態ごとの生徒数をカウント)
state_counts = defaultdict(int)

# 生徒ごとのテスト結果を元に状態に生徒をカウント
for result in dataset:
    state_tuple = tuple(map(int, result))  # np.int64をint型に変換
    state_counts[state_tuple] += 1

# 各状態の生徒数を表示
for state, count in state_counts.items():
    formatted_state = ''.join(map(str, state))  # 状態を'01010'のような形式に変換
    print(f"State [{formatted_state}]: {count} students")


# 遷移確率を計算するための準備
transition_probabilities = defaultdict(dict)

# 各状態について、次の状態に遷移する確率を計算
for current_state, count in state_counts.items():
    # 次に解ける可能性のある未解決の問題を見つける
    for i in range(num_questions):
        if current_state[i] == 0:  # まだ解けていない問題
            # その問題が解けた状態を生成（遷移後の状態）
            next_state = list(current_state)
            next_state[i] = 1
            next_state = tuple(next_state)
            
            # 現在の状態から次の状態に遷移する確率を仮定 (生徒が1問解ける)
            transition_probabilities[current_state][next_state] = 1 / (num_questions - sum(current_state))

# 遷移確率の表示
for current_state, next_states in transition_probabilities.items():
    print(f"From state {current_state}:")
    for next_state, prob in next_states.items():
        print(f"  To {next_state} with probability {prob:.2f}")