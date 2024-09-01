import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
n_embed = 65
#n_heads = 4
block_size = 32
lr = 1e-2
dropout = 0.1

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



def get_batch(split, batch_size=16):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0,len(data)-block_size,(batch_size,))
    x = torch.stack([data[index:index+block_size] for index in ix])
    y = torch.stack([data[index+1:index+block_size+1] for index in ix])
    return x,y

# what does head do?
# with Q,K,V matrices we use the calculations to obtain the probabilites Z that have same shape as V.
# nn.linear takes args: # features, # neurons.
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.Wq = nn.Linear(n_embed,head_size, bias = False) # W = (c, head_size) because XW -> b,t,c @ c,head_size -> b,t,head_size
        self.Wk = nn.Linear(n_embed,head_size, bias = False)
        self.Wv = nn.Linear(n_embed,head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B,T,C = x.shape # C is now head_size = # features per char = 8
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        attn = q @ k.transpose(-2,-1) * (C**-0.5)          #b,t,c @ b,c,t just want to switch last 2 dims of K. attn.shape -> b,t,t
        #print(self.tril.shape, attn.shape)
        attn = attn.masked_fill(self.tril[:T][:T] == 0, float('-inf')) # we index through with :T,:T in case our input batch has sequences smaller than block_size
        attn = F.softmax(attn,dim=-1)
        self_attn = attn @ v       # b,t,t @ b,t,c -> b,t,c
       # print(self_attn.shape)
        return self_attn

# just concatenates the last set of vectors in self-attention output matrix Z. if we want the length of vectors representing chars to be n_embed, we must make 
# head_size = n_embed // n_heads because at the end we are multiplying head_size by n_heads
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(n_heads)] # 32,8,16*4 -> where do i put batch size and context length together into 256?
        self.proj = nn.Linear(n_embed-1, n_embed) # 64,65
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads] , dim=-1) # x = (batch_size,context_length , n_embed)
        out = self.proj(out) # x@w = x.shape @ n_embed, n_embed = batch_size, context_length, n_embed = 32,8,65
        out = self.dropout(out)
        return out

# works on each char individually. Applied to matrix Z (self-attention output) -> (batch_size, context_length, n_embed) @ (n_embed, n_embed(could be diff.))
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.mlp(x)
    

class Block(nn.Module):
    
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads # -> 65  /    4   ->   16
        self.self_attn = MaskedMultiHeadAttention(n_heads, head_size)
        self.feedforward = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.self_attn(self.ln1(x))
        out = x + self.feedforward(self.ln2(x))
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.blocks = nn.Sequential(
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4),
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        B,T = x.shape
        x = self.token_embedding_table(x) + self.position_embedding_table(torch.arange(T)) # positional embeddings = number each char 0-7 lol
        x = self.blocks(x)
        logits = self.lm_head(x)

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
            chars_enc_cond = chars_enc[:,-block_size:] # just want last block_size num of chars for context. as we generate chars, context will increase infinitely
            logits, _ = self(chars_enc_cond)
            next_char_probs = logits[:,-1,:]
            next_char_probs = F.softmax(next_char_probs, dim=1)
            next_char = torch.multinomial(next_char_probs, num_samples=1)
            chars_enc = torch.cat((chars_enc, next_char), dim=1)
            
        return chars_enc
    

# training
model = BigramLanguageModel()
optimizer = optim.AdamW(model.parameters(),lr)
eval_interval = 500
max_iters = 10000
for i in range(max_iters):
    # if max_iters % 2 == 0:
    #     optimizer.param_groups[0]['lr'] = 1e-3
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,block_size),dtype=torch.long) # this tensor column must match block_size
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))