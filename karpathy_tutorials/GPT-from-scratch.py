import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

block_size = 8
with open('tinyshakespeare.txt', 'r', encoding = 'utf-8') as file:
    text = file.read()
all_chars = sorted(list(set(text)))
all_chars = ''.join(all_chars)
vocab_size = len(all_chars)
stoi = {s:i for i,s in enumerate(all_chars)}
itos = {i:s for i,s in enumerate(all_chars)}
encode = lambda x: [stoi[char] for char in x]
decode = lambda x: ''.join(itos[i] for i in x)
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

eval_iters = 200
@torch.no_grad()
def estimate_loss():
    model.eval()
    lossi_train = torch.zeros(eval_iters)
    lossi_valid = torch.zeros(eval_iters) 
    for i in range(eval_iters):
        xb_train,yb_train = get_batch('train', 32)
        xb_valid,yb_valid = get_batch('valid', 32)
        _, loss_train = model(xb_train,yb_train)
        _, loss_valid = model(xb_valid,yb_valid)
        lossi_train[i] = loss_train
        lossi_valid[i] = loss_valid
    model.train()
    return lossi_train.mean(),lossi_valid.mean()



def get_batch(split, batch_size=32):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0,len(data)-block_size,(batch_size,))
    x = torch.stack([data[index:index+block_size] for index in ix])
    y = torch.stack([data[index+1:index+block_size+1] for index in ix])
    return x,y

class BigramLanguageModel(nn.Module):
    def __init__(self, n_embd=None):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.token_embedding_table(x)

        if targets is None:
            return logits, None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
    def generate(self, chars_enc, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(chars_enc)
            next_char_probs = logits[:,-1,:]
            next_char_probs = F.softmax(next_char_probs, dim=1)
            next_char = torch.multinomial(next_char_probs, num_samples=1)
            chars_enc = torch.cat((chars_enc, next_char), dim=1)
            
        return chars_enc
model = BigramLanguageModel()
optimizer = optim.AdamW(model.parameters(),1e-3)
eval_interval = 200
for i in range(10000):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(torch.zeros((1,1),dtype=torch.long), 300)[0].tolist()))