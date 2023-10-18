
import torch
import torch.nn as nn
import math
import json
import torch_dct as dct
from torch.utils.checkpoint import checkpoint


def attn_selector(attn_type, config, W_q=None, W_k=None, W_v=None):


    if attn_type.startswith("softmax"):
        attn = SoftmaxAttention(config)

    elif attn_type.startswith("mlp"):
        attn = MLPAttention(config)

    elif attn_type.startswith("concat"):
        attn = MLPConCat(config)

    elif attn_type.startswith("encode"):
        attn = MLPEncode(config)
        
    elif attn_type.startswith("dct"):
        attn = DCTAttention(config)
        
    elif attn_type.startswith("conv"):
        attn = ConvAttention(config)

    elif attn_type.startswith("kernelized"):
        attn = SoftmaxAttention_RBF(config)

    elif attn_type.startswith("linformer"):
        from models.attention_linformer import LinformerAttention
        attn = LinformerAttention(config)
    elif attn_type.startswith("informer"):
        from models.attention_informer import ProbAttention
        attn = ProbAttention(config)
    elif attn_type.startswith("nystrom"):
        from models.attention_nystrom import NystromAttention
        attn = NystromAttention(config)
    elif attn_type.startswith("performer"):
        from models.attention_performer import PerformerAttention
        attn = PerformerAttention(config)
    elif attn_type.startswith("bigbird"):
        from models.attention_bigbird import BigBirdAttention
        attn = BigBirdAttention(config)
    elif attn_type.startswith("reformer"):
        from models.attention_reformer import LSHAttention
        attn = LSHAttention(config, W_q, W_k, W_v)
    elif attn_type.startswith("skyformer"):
        from models.attention_skyformer import Skyformer
        attn = Skyformer(config)

    return attn


class SoftmaxAttention_RBF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        # input [batch_size, nb_heads, seq_len, dim_head]
        
        b,h,n,p = Q.shape
        
        data_normalizer = (p ** -0.25)
        Q = Q * (mask[:, None, :, None] * data_normalizer)
        K = K * (mask[:, None, :, None] * data_normalizer)
        # v = v * mask[:, None, :, None]
        
        diag_Q = (Q * Q).sum(-1) * 0.5
        diag_Q = diag_Q.unsqueeze(dim=-1)
        diag_K = (K * K).sum(-1) * 0.5
        diag_K = diag_K.unsqueeze(dim=-2)


        product = (torch.einsum('...np,...mp->...nm', Q, K) - diag_Q 
            - diag_K) - 1e9 * (1 - mask[:, None, None, :])
        attn = torch.exp(product)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        # output [batch_size, nb_heads, seq_len, dim_head]
        return X


class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        # input [batch_size, nb_heads, seq_len, dim_head]
        # print('Q', Q.abs().median()) # check scale
        # print('K', K.abs().median())
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)


        X = torch.matmul(attn, V)

        # output [batch_size, nb_heads, seq_len, dim_head]
        return X

class DCTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, X, mask=None):

        y = dct.dct(X)
        attn_out = dct.dct(y, 1)
        # output [batch_size, seq_len, dim]

        return attn_out


class ConvAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.dim = config["transformer_dim"] # input_dim
        self.num_head = config["num_head"]
        self.attn_type = config["attn_type"]
        self.seq_len = config["max_seq_len"]
        # self.hidden_size = self.seq_len
        self.hidden_size = config["hidden_size"]
        self.kernel_size = config["kernel_size"]
        
        self.W_x = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.nnet = nn.Sequential(
        nn.Conv1d(self.num_head*self.head_dim, self.hidden_size, kernel_size=self.kernel_size, groups=self.num_head, padding='same'),
        # nn.ReLU(),
        # nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Conv1d(self.hidden_size, self.num_head*self.seq_len, kernel_size=self.kernel_size, groups=self.num_head, padding='same'),
        )


    def forward(self, X, mask):

        # input [batch_size, seq_len, dim]

        V = self.split_heads(self.W_v(X)) # [batch_size, seq_len, num_head * head_dim]

        X = self.W_x(X)  # [batch_size, seq_len, num_head * head_dim]

        X = X.permute(0, 2, 1) # [batch_size, num_head * head_dim, seq_len]

        wei = self.nnet(X)  # [batch_size, nb_heads*seq_len, seq_len]

        # Get the dimensions of the input tensor
        batch_size, product_dim, seq_len = wei.size()

        # Calculate the number of heads
        nb_heads = product_dim // seq_len

        # Reshape the input tensor
        wei = wei.view(batch_size, nb_heads, seq_len, seq_len)
        
        
        wei = wei - 1e6 * (1 - mask[:, None, None, :])

        wei = nn.functional.softmax(wei, dim = -1)
        wei = self.drop_attn(wei)

      
        attn_out = torch.matmul(wei, V) # [batch_size, nb_heads, seq_len, seq_len] * [batch_size, nb_heads, seq_len, dim_head] -> [batch_size, nb_heads, seq_len, dim_head] 

        attn_out = self.combine_heads(attn_out)

        # output [batch_size, seq_len, dim]
        return attn_out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

#one head one batch
class MLPEncode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.dim = config["transformer_dim"] # input_dim
        self.attn_type = config["attn_type"]
        self.seq_len = config["max_seq_len"]
        self.hidden_size = config["hidden_size"]
        
        self.linear_V = nn.Linear(self.dim, self.dim)
        self.attention_net = nn.Sequential(
            nn.Linear(2*self.dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.seq_len)
        )

    def forward(self, X, mask):
        X = X.view(self.seq_len, self.dim)
        V = self.linear_V(X)

        
        P0 = self.positional_encoding(self.dim, self.seq_len).cuda()
        E = P0*X
        weighted_sum_tensor = torch.zeros(self.seq_len, self.dim).cuda()

        # Calculate the weighted sum for each row
        for i in range(self.seq_len):
            # Weight all rows except the current row equally
            weighted_rows = torch.cat((E[:i], E[i+1:]), dim=0)
            uniform_weights = torch.ones(weighted_rows.shape[0]).cuda()
            weighted_sum = torch.sum(weighted_rows * uniform_weights.view(-1, 1), dim=0)
            weighted_sum_tensor[i] = weighted_sum/(X.shape[0]-1)

        # Concatenate the X and weighted_sum_tensor along dim=1
        result_tensor = torch.cat((X, weighted_sum_tensor), dim=1)

        att_scores = self.attention_net(result_tensor)
        # att_scores = att_scores.view(self.seq_len, self.seq_len)
        # wei = wei - 1e6 * (1 - mask[:, None, None, :]) NOT IMPLEMENTED
        att_weights = torch.softmax(att_scores, dim=-1)
        att_weights = self.drop_attn(att_weights)
        weighted_sum = torch.matmul(att_weights, V)

        return weighted_sum

    def positional_encoding(self, d_model, max_sequence_length):
      even_i = torch.arange(0, d_model, 2).float().cuda()
      denominator = torch.pow(10000, even_i / d_model)
      position = torch.arange(max_sequence_length).reshape(max_sequence_length, 1).cuda()
      even_PE = torch.sin(position / denominator)
      odd_PE = torch.cos(position / denominator)
      stacked = torch.stack([even_PE, odd_PE], dim=2)
      PE = torch.flatten(stacked, start_dim=1, end_dim=2)
      return PE

class MLPConCat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.dim = config["transformer_dim"] # input_dim
        self.attn_type = config["attn_type"]
        self.seq_len = config["max_seq_len"]
        self.hidden_size = config["hidden_size"]
        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        
        self.W_x = nn.Linear(self.dim, self.num_head*self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head*self.head_dim)
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.seq_len*self.head_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.seq_len*self.seq_len)
        )

    def forward(self, X, mask):
        V = self.split_heads(self.W_v(X)) 
        X = self.split_heads(self.W_x(X)) 
        P0 = X.reshape(-1, self.num_head, self.seq_len * self.head_dim)
        att_scores = self.attention_net(P0)
        att_scores = att_scores.reshape(-1, self.num_head, self.seq_len, self.seq_len)
        # wei = wei - 1e6 * (1 - mask[:, None, None, :]) NOT IMPLEMENTED
        att_weights = torch.softmax(att_scores, dim=-1)
        att_weights = self.drop_attn(att_weights)
        weighted_sum = torch.matmul(att_weights, V)
        weighted_sum = self.combine_heads(weighted_sum)
        return weighted_sum

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return 



# #one head one batch
# class MLPConCat(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
#         self.dim = config["transformer_dim"] # input_dim
#         self.attn_type = config["attn_type"]
#         self.seq_len = config["max_seq_len"]
#         self.hidden_size = config["hidden_size"]
        
#         self.linear_V = nn.Linear(self.dim, self.dim)
#         self.attention_net = nn.Sequential(
#             nn.Linear(self.seq_len*self.dim, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.seq_len*self.seq_len)
#         )

    # def forward(self, X, mask):
    #     X = X.view(self.seq_len, self.dim)
    #     V = self.linear_V(X)
    #     P0 = X.view(-1, self.seq_len * self.dim)
    #     att_scores = self.attention_net(P0)
    #     att_scores = att_scores.view(self.seq_len, self.seq_len)
    #     # wei = wei - 1e6 * (1 - mask[:, None, None, :]) NOT IMPLEMENTED
    #     att_weights = torch.softmax(att_scores, dim=-1)
    #     att_weights = self.drop_attn(att_weights)
    #     weighted_sum = torch.matmul(att_weights, V)

    #     return weighted_sum

class MLPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.dim = config["transformer_dim"] # input_dim
        self.num_head = config["num_head"]
        self.attn_type = config["attn_type"]
        self.seq_len = config["max_seq_len"]
        # self.hidden_size = self.seq_len
        self.hidden_size = config["hidden_size"]

        self.W_x = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.nnet = nn.Sequential(
        nn.Linear(self.head_dim, self.hidden_size),
        # nn.ReLU(),
        # nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.seq_len, bias=False),
        )


    def forward(self, X, mask):

        # input [batch_size, seq_len, dim]


        
        V = self.split_heads(self.W_v(X)) # [batch_size, nb_heads, seq_len, dim_head]



        X = self.split_heads(self.W_x(X))  # [batch_size, nb_heads, seq_len, dim_head]



        wei = self.nnet(X)  # [batch_size, nb_heads, seq_len, seq_len]

        wei = wei - 1e6 * (1 - mask[:, None, None, :])

        wei = nn.functional.softmax(wei, dim = -1)
        wei = self.drop_attn(wei)


        attn_out = torch.matmul(wei, V) # [batch_size, nb_heads, seq_len, seq_len] * [batch_size, nb_heads, seq_len, dim_head] -> [batch_size, nb_heads, seq_len, dim_head] 

        attn_out = self.combine_heads(attn_out)
 
        # output [batch_size, seq_len, dim]
        return attn_out
    

    
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.dim = config["transformer_dim"] # input_dim
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = attn_selector(self.attn_type, config, self.W_q, self.W_k, self.W_v)

        self.grad_checkpointing = (self.attn_type == "softmax")

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer") or self.attn_type.startswith("mlp") or self.attn_type.startswith("conv") or self.attn_type.startswith("dct") or self.attn_type.startswith("concat") or self.attn_type.startswith("encode"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())

        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))
            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)
            
        # if not self.attn_type.startswith("concat"):
        out = self.ff(attn_out)
        # else:
        #     out = attn_out

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


