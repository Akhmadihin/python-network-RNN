import numpy as np #pip install numpy
import random #pip install random2

# 1. Загрузка текста
with open('train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Создание словаря
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# 3. Параметры
hidden_size = 128
seq_length = 40
learning_rate = 0.01

# 4. Инициализация весов
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def forward(inputs, h_prev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)
    
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = softmax(ys[t]).flatten()  # Критическое изменение
    
    return xs, hs, ps

def backward(xs, hs, ps, targets):
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])
    
    for t in reversed(range(len(xs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        
        dWhy += np.dot(dy.reshape(-1, 1), hs[t].T)
        dby += dy.reshape(-1, 1)
        
        dh = np.dot(Why.T, dy.reshape(-1, 1)) + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh
        
        dbh += dh_raw
        dWxh += np.dot(dh_raw, xs[t].T)
        dWhh += np.dot(dh_raw, hs[t-1].T)
        dh_next = np.dot(Whh.T, dh_raw)
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    
    return dWxh, dWhh, dWhy, dbh, dby

# Обучение
n_iterations = 1000
for i in range(n_iterations):
    start_idx = random.randint(0, len(text) - seq_length - 1)
    inputs = [char_to_idx[ch] for ch in text[start_idx:start_idx + seq_length]]
    targets = [char_to_idx[ch] for ch in text[start_idx+1:start_idx + seq_length + 1]]
    
    h_prev = np.zeros((hidden_size, 1))
    xs, hs, ps = forward(inputs, h_prev)
    dWxh, dWhh, dWhy, dbh, dby = backward(xs, hs, ps, targets)
    
    for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
        param -= learning_rate * dparam
    
    if i % 500 == 0:
        sample_idx = random.randint(0, len(inputs) - 1)
        current_loss = -np.log(ps[sample_idx][targets[sample_idx]])
        print(f"Iteration {i}, Loss: {current_loss:.4f}")

def generate_text(seed, length=200):
    h = np.zeros((hidden_size, 1))
    idxes = [char_to_idx[ch] for ch in seed]
    
    for _ in range(length):
        x = np.zeros((vocab_size, 1))
        x[idxes[-1]] = 1
        
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = softmax(y).flatten()
        
        next_idx = np.random.choice(range(vocab_size), p=p)
        idxes.append(next_idx)
    
    return ''.join([idx_to_char[idx] for idx in idxes])

print(generate_text("Привет", 500))
