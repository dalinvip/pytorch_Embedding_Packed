import torch
from torch.autograd import Variable
from Embedding import Embedding

# random pretrain_embedding
pretrain_embedding = torch.randn(10, 128)

# multiple nn.Embedding()
Embed = Embedding(num_embeddings=[10, 15], embedding_dim=[128, 256], padding_idx=[None, None],
                  pretrained_embedding=[pretrain_embedding, None], init_weight=True)

# test embedding
# input is batch_size * length * num_features, num_features means that multiple embedding features
input = Variable(torch.ones(16, 18, 2).type(torch.LongTensor))  # batch_size * length * num_features

# concat the multiple features embedding after embedding and output
emb = Embed.after_embedding(input)
print(emb.size())