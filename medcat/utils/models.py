import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class LSTM(nn.Module):
    def __init__(self, embeddings, padding_idx, nclasses=2, bid=True, input_size=300,
                 num_layers=2, hidden_size=300, dropout=0.5, dependencies=None):
        super(LSTM, self).__init__()
        self.padding_idx = padding_idx
        # Get the required sizes
        vocab_size = len(embeddings)
        embedding_size = len(embeddings[0])

        self.num_layers = num_layers
        self.bid = bid
        self.input_size = input_size
        self.nclasses = nclasses
        self.num_directions = (2 if self.bid else 1)
        self.dropout = dropout

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embeddings.load_state_dict({'weight': embeddings})
        # Disable training for the embeddings - IMPORTANT
        self.embeddings.weight.requires_grad = False

        self.hidden_size = hidden_size
        bid = True # Is the network bidirectional

        # Create the RNN cell - devide 
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size // self.num_directions,
                           num_layers=self.num_layers,
                           dropout=dropout,
                           bidirectional=self.bid)
        self.fc1 = nn.Linear(self.hidden_size, nclasses)

        self.d1 = nn.Dropout(dropout)


    #
    #     dependencies_bilstm = nn.LSTM(input_size=73, hidden_size=16, num_layers=2, dropout=0.5, bidirectional=True)
    #     left = torch.randn((1, 20))  # random input vector as an example input
    #     right = left.unsqueeze(0)
    #     dependencies2 = dependencies.unsqueeze(0)
    #
    #     (dependencies_bilstm_hidden_states, _) = dependencies_bilstm(dependencies2)
    # #    concatenated_bilstm_hidden_states = torch.cat((dependencies_bilstm_hidden_states, self.rnn), dim=2)


    def forward(self, input_ids, center_positions, attention_mask=None, ignore_cpos=False):
        x = input_ids
        #Get the mask from x
        if attention_mask is None:
            mask = x != self.padding_idx
        else:
            mask = attention_mask

        # Embed the input: from id -> vec
        x = self.embeddings(x) # x.shape = batch_size x sequence_length x emb_size

        # Tell RNN to ignore padding and set the batch_first to True
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1).cpu(), batch_first=True, enforce_sorted=False)

        # Run 'x' through the RNN
        x, hidden = self.rnn(x)

        # Add the padding again
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Get what we need
        row_indices = torch.arange(0, x.size(0)).long()

        # If this is  True we will always take the last state and not CPOS
        if ignore_cpos:
            x = hidden[0]
            x = x.view(self.num_layers, self.num_directions, -1, self.hidden_size//self.num_directions)
            x = x[-1, :, :, :].permute(1, 2, 0).reshape(-1, self.hidden_size)
        else:
            x = x[row_indices, center_positions, :]

        # Push x through the fc network and add dropout
        x = self.d1(x)
        x = self.fc1(x)

        return x


class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs):
        outputs = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, outputs


class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        emb_matrix = torch.Tensor(emb_matrix)
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(args.token_vocab_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)

        self.pos_emb = nn.Embedding(args.pos_vocab_size, args.pos_dim,
                                    padding_idx=0) if args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(args.post_vocab_size, args.post_dim,
                                     padding_idx=0) if args.post_dim > 0 else None  # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)

        # gcn layer
        self.gcn = GCN(args, embeddings, args.hidden_dim, args.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, post, mask, l = inputs  # unpack inputs
        maxlen = max(l.data)

        def inputs_to_tree_reps(head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=self.args.loop).reshape(1, maxlen,
                                                                                                          maxlen) for
                   tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda()

        adj = inputs_to_tree_reps(head.data, tok.data, l.data)
        h = self.gcn(adj, inputs)

        # avg pooling asp feature
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h
        outputs = (h * mask).sum(dim=1) / asp_wn  # mask h

        return outputs


class GCN(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, args.rnn_hidden, 1, batch_first=True, bidirectional=True)
        self.in_dim = args.rnn_hidden * 2

        # drop out
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        tok, asp, pos, head, post, mask, l = inputs  # unpack inputs
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        gcn_inputs = self.encode_with_rnn(embs, l, tok.size()[0])

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1  # norm
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()



class BertForMetaAnnotation(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        center_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (batch_size, sequence_length, hidden_size)

        row_indices = torch.arange(0, sequence_output.size(0)).long()
        sequence_output = sequence_output[row_indices, center_positions, :]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
