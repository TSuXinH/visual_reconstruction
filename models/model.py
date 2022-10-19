import torch.nn as nn
import torch
from torch.nn.modules import dropout


# Fully connected neural network
class FCN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCN, self).__init__()
        self.output_size = output_size
        self.i2o = nn.Sequential(
            nn.Linear(input_size,2000),
            nn.ReLU(),
            nn.Linear(2000,500),
            nn.ReLU(),
            nn.Linear(500,2000),
            nn.ReLU(),
            nn.Linear(2000, self.output_size[0]*self.output_size[1])
        )
        
    def forward(self, input):
        out = self.i2o(input)
        out = out.reshape(-1,1,self.output_size[0],self.output_size[1])
        return out

# Fully connected neural network
class FCN_ae(nn.Module):
    def __init__(self, input_size, output_size,time_wind=1):
        super(FCN_ae, self).__init__()
        self.output_size = output_size
        self.i2o = nn.Sequential(
            # nn.Conv1d(time_wind,1,1,padding='same'),
            nn.Linear(input_size,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,200),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(200, self.output_size)
        )
        
    def forward(self, input):
        out = self.i2o(input)
        return out

# Fully connected neural network
class FCN_ae_debug(nn.Module):
    def __init__(self, input_size, output_size,time_wind=1):
        super(FCN_ae_debug, self).__init__()
        self.output_size = output_size
        self.i2o = nn.Sequential(
            # nn.Conv1d(time_wind,1,1,padding='same'),
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_size)
        )
        
    def forward(self, input):
        out = self.i2o(input)
        return out

# RNN network
class RNN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size=500):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2o = nn.Sequential(
            nn.Linear(hidden_size*2,500),
            nn.ReLU(),
            nn.Linear(500,2000),
            nn.ReLU(),
            nn.Linear(2000, self.output_size[0]*self.output_size[1])
        )
        self.i2h = nn.Sequential(
            nn.Linear(input_size,2000),
            nn.ReLU(),
            nn.Linear(2000,hidden_size),
            nn.ReLU(),
        )
        
    
    def forward(self, input,hidden):
        hidden_new = self.i2h(input)
        combined = torch.cat((hidden_new, hidden), 1)
        output = self.i2o(combined)
        # output = torch.clamp(output,0,255)
        output = output.reshape(-1,1,self.output_size[0],self.output_size[1])
        return output,hidden
    
    def initHidden(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size)

class GRU_ae(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50,num_layers=3,device='cuda'):
        super(GRU_ae, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True,dropout=0.2)
        self.fcn = nn.Linear(hidden_size,output_size)
        self.to(device)

    def forward(self,inputs):
        output,hn = self.gru(inputs)
        output = self.fcn(output)
        return output[:,-1:]



# RNN network
class RNN_ae(nn.Module):
    def __init__(self, input_size, output_size,hidden_size=500,device='cuda'):
        super(RNN_ae, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.i2o = nn.Sequential(
            nn.Linear(hidden_size*2,500),
            nn.ReLU(),
            nn.Linear(500,200),
            nn.ReLU(),
            nn.Linear(200, self.output_size)
        )
        self.i2h = nn.Sequential(
            nn.Linear(input_size,100),
            nn.ReLU(),
            nn.Linear(100,hidden_size),
            nn.ReLU(),
        )
        self.to(self.device)
        
    
    def _forward(self, input,hidden):
        hidden_new = self.i2h(input)
        combined = torch.cat((hidden_new, hidden), 1)
        output = self.i2o(combined)
        return output,hidden

    def forward(self,inputs):
        N,C = inputs.size()[:2]
        hid = self.initHidden(C)
        for i in range(N):
            out,hid = self._forward(inputs[i],hid)
        return out

    
    def initHidden(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size).to(self.device)

# transformer
######################################################################
# Seq2Seq Network using Transformer
# ---------------------------------
#
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks. 
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the 
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model. 
# Finally, the output of Transformer model is passed through linear layer
# that give un-normalized probabilities for each token in the target language. 
#


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import einops
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 20):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).to(DEVICE)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:,:token_embedding.size(1)])

class LearnPosEncoding(nn.Module):
    def __init__(self,frames,emb_size):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, frames, emb_size))

    def forward(self, token_embedding: Tensor):
        return token_embedding+self.pos_embedding

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, input_size: int, emb_size:int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size,2000),
            nn.ReLU(),
            nn.Linear(2000,emb_size),
        ).to(DEVICE)

    def forward(self, input: Tensor):
        return self.embedding(input)


# Seq2Seq Network 
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_len: int,
                 tgt_size,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,batch_first=True).to(DEVICE)
        tgt_len = tgt_size[0]*tgt_size[1]
        self.bof_emb = nn.Parameter(torch.randn(1, 1, emb_size)).to(DEVICE)
        self.generator = nn.Linear(emb_size, tgt_len).to(DEVICE)
        self.src_tok_emb = TokenEmbedding(src_len, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_len, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def create_mask(self,src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

        return src_mask, tgt_mask

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_padding_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=None):
        N,C,H,W = tgt.shape
        tgt = tgt.reshape(N,C,-1)
        bof_emb = einops.repeat(self.bof_emb, '() n d -> b n d', b=N)
        src_emb = self.positional_encoding(torch.cat((bof_emb,self.src_tok_emb(src)),dim=1))
        tgt_emb = self.positional_encoding(torch.cat((bof_emb,self.tgt_tok_emb(tgt)),dim=1))

        src_mask,tgt_mask = self.create_mask(src_emb,tgt_emb)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        outs = self.generator(outs)
        return outs.reshape(N,C+1,H,W)

    def encode(self, src: Tensor, src_mask: Tensor):
        N,C,D = src.shape
        bof_emb = einops.repeat(self.bof_emb, '() n d -> b n d', b=N)
        return self.transformer.encoder(self.positional_encoding(
                            torch.cat((bof_emb,self.src_tok_emb(src)),dim=1)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        N,C,H,W = tgt.shape
        tgt = tgt.reshape(N,C,-1)
        bof_emb = einops.repeat(self.bof_emb, '() n d -> b n d', b=N)
        out = self.transformer.decoder(self.positional_encoding(
                          torch.cat((bof_emb,self.tgt_tok_emb(tgt)),dim=1)), memory,
                          tgt_mask)
        return out

######################################################################
# During training, we need a subsequent word mask that will prevent model to look into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both. 
#


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, tgt_shape, start_symbol=0):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    N,C,H,W = tgt_shape
    memory = model.encode(src, src_mask)
    ys = torch.ones(N,1,H,W).fill_(start_symbol).type(torch.float32).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1)+1)
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        next_y = model.generator(out[:,-1]).reshape(N,1,H,W)

        ys = torch.cat([ys,next_y], dim=1)
    return ys

# actual function to translate input neural data into target image
def translate(model: torch.nn.Module, src,tgt_shape):
    model.eval()
    num_frames = src.shape[1]
    src_mask = (torch.zeros(num_frames, num_frames)).type(torch.bool)
    tgt = greedy_decode(model, src, src_mask, max_len=num_frames,tgt_shape=tgt_shape)
    return tgt

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    return src_mask, tgt_mask


# Seq2Single Network 
class Seq2SingleTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, src_size,tgt_size, emb_size, nhead, nhid, nlayers,frames=11, dropout=0.1):
        super(Seq2SingleTransformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_encoder = LearnPosEncoding(frames+1,emb_size)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = TokenEmbedding(src_size, emb_size)
        self.emb_size = emb_size
        self.tgt_size = tgt_size
        tgt_len = tgt_size[0]*tgt_size[1]
        self.decoder = nn.Linear(emb_size,tgt_len)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        N,C,D = src.shape
        C=C+1
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != C:
                mask = self._generate_square_subsequent_mask(C).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.emb_size)
        cls_token = einops.repeat(self.cls_token, '() n d -> b n d', b=N)
        src = torch.cat((src,cls_token),dim=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output.reshape(N,C,self.tgt_size[0],self.tgt_size[1])