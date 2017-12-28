# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class Embedding():
    def __init__(self, num_embeddings, embedding_dim, padding_idx, pretrained_embedding, init_weight=True,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):

        """
        :param num_embeddings: num_embeddings (tuple): size of the dictionary of embeddings for different nn.embedding
        :param embedding_dim: embedding_dim (tuple): the size of each embedding vector for different nn.embedding
        :param padding_idx: (tuple): If given, pads the output with zeros for different nn.embedding whenever it
                                     encounters the index. If not given, please set it is None
        :param pretrained_embedding: pretrained_embedding (tuple/None): loading pretrained word embedding
                                      pretrained_embedding=None:not use pretrained word embedding for all
        :param init_weight:(boolean, optional): initial embedding weigth by formula, default is True
        :param max_norm:(float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        :param norm_type:(float, optional): The p of the p-norm to compute for the max_norm option
        :param scale_grad_by_freq(boolean, optional):if given, this will scale gradients by the frequency of
                                                the words in the mini-batch.
        :param sparse:(boolean, optional): if ``True``, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.
        """

        assert isinstance(num_embeddings, list), "Error, the type of num_embeddings must be list"
        assert isinstance(embedding_dim, list), "Error, the type of embedding_dim must be list"
        assert isinstance(padding_idx, list), "Error, the type of padding_idx must be list"
        if pretrained_embedding is not None:
            assert isinstance(pretrained_embedding, list), "Error, the type of pretrained_embedding must be list"

        # assert length
        if pretrained_embedding is None:
            assert len(num_embeddings) == len(embedding_dim) == len(padding_idx), \
                "Error, the length of num_embeddings/embedding_dim/padding_idx must be same if you use list for " \
                "the padding_idx and pretrained_embedding"
        else:
            assert len(num_embeddings) == len(embedding_dim) == len(padding_idx) == len(pretrained_embedding), \
                "Error, the length of num_embeddings/embedding_dim/padding_idx/pretrained_embedding must be same " \
                "if you use list for the padding_idx and pretrained_embedding"

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.pretrained_embedding=pretrained_embedding
        self.init_weight = init_weight
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.num_feature = len(self.num_embeddings)
        self.Embedding = []

        self.embed()

        if self.pretrained_embedding is not None and len(pretrained_embedding) >= 1:
            self.pretrained_Embedding()

        if self.init_weight is True:
            self.initial_weight()

    def embed(self):
        """
        :return: all features embedding like [Embedding(10, 128), Embedding(15, 256, padding_idx=2)]
        """
        for embed in range(self.num_feature):
            self.Embedding.append(self.create_embedding(num_embeddings=self.num_embeddings[embed],
                                                        embedding_dim=self.embedding_dim[embed],
                                                        padding_idx=self.padding_idx[embed]))
        return self.Embedding

    def create_embedding(self, num_embeddings, embedding_dim, padding_idx):
        """
        :param num_embeddings: num_embeddings (tuple): size of the dictionary of embeddings for different nn.embedding
        :param embedding_dim: embedding_dim (tuple): the size of each embedding vector for different nn.embedding
        :param padding_idx: (tuple): If given, pads the output with zeros for different nn.embedding whenever it
                                     encounters the index. If not given, please set it is None
        :return: embedding like Embedding(15, 256, padding_idx=2)
        """
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                 padding_idx=padding_idx, max_norm=self.max_norm, norm_type=self.norm_type,
                                 scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)
        return embedding

    def pretrained_Embedding(self):
        """
        :Function: loading pretrain word embedding
        """
        print("loading pretrained embedding")
        for index, pretrain_embedding in enumerate(self.pretrained_embedding):
            if pretrain_embedding is None:
                continue
            pretrained_embedding_weight = np.array(pretrain_embedding)
            self.Embedding[index].weight.data.copy_(torch.from_numpy(pretrained_embedding_weight))

    def initial_weight(self):
        """
         :Function: initial weight for embedding attribute of weights
        :return:
        """
        print("initial weight")
        for index, embedding in enumerate(self.Embedding):
            value = np.sqrt(3 / self.embedding_dim[index])
            init.uniform(embedding.weight, -value, value)

    def after_embedding(self, input):
        """
        :Function: for the input of integer, convert to weight value
        :param input: input size : batch_size * sentence_length * number_features
        :return: the result of multiple embedding concat , size :  batch_size * sentence_length * (h1 + h2 + h3 + ...)
        """
        if input.data.dim() is not 3:
            print("Error, expect the dim of input is 3")
            exit()
        if input.size(2) is not len(self.Embedding):
            print("Error, the number feature of input must be same with the Embedding")
            exit()

        print("input size {}".format(input.size()))

        input = input.permute(2, 0, 1)
        embedding_concat = []
        for num_feature in range(input.size(0)):
            embed = self.Embedding[num_feature](input[num_feature])
            embedding_concat.append(embed)
        output = torch.cat(embedding_concat, 2)

        return output


