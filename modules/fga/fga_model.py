import torch
import torch.nn as nn
import torch.nn.functional as F
from atten import Atten


class FGA(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, hidden_ques_dim, hidden_ans_dim,
                 hidden_hist_dim, hidden_cap_dim, hidden_img_dim):
        '''
        Factor Graph Attention
        :param vocab_size: vocabulary size
        :param word_embed_dim
        :param hidden_ques_dim:
        :param hidden_ans_dim:
        :param hidden_hist_dim:
        :param img_features_dim:
        '''
        super(FGA, self).__init__()

        print("Init FGA with vocab size %s, word embed %s, hidden ques %s, hidden ans %s,"
              " hidden hist %s, hidden cap %s, hidden img %s" % (vocab_size, word_embed_dim,
                                                                hidden_ques_dim,
                                                                hidden_ans_dim,
                                                                hidden_hist_dim,
                                                                hidden_cap_dim,
                                                                hidden_img_dim))
        self.hidden_ques_dim = hidden_ques_dim
        self.hidden_ans_dim = hidden_ans_dim
        self.hidden_cap_dim = hidden_cap_dim
        self.hidden_img_dim = hidden_img_dim
        self.hidden_hist_dim = hidden_hist_dim

        # Vocab of History LSTMs is one more as we are keeping a stop id (the last id)
        self.word_embedddings = nn.Embedding(vocab_size+1+1, word_embed_dim, padding_idx=0)

        self.lstm_ques = nn.LSTM(word_embed_dim, self.hidden_ques_dim, batch_first=True)
        self.lstm_ans = nn.LSTM(word_embed_dim, self.hidden_ans_dim, batch_first=True)

        self.lstm_hist_ques = nn.LSTM(word_embed_dim, self.hidden_hist_dim, batch_first=True)
        self.lstm_hist_ans = nn.LSTM(word_embed_dim, self.hidden_hist_dim, batch_first=True)

        self.lstm_hist_cap = nn.LSTM(word_embed_dim, self.hidden_cap_dim, batch_first=True)


        self.qahistnet = nn.Sequential(
            nn.Linear(self.hidden_hist_dim*2, self.hidden_hist_dim),
            nn.ReLU(inplace=True)
        )

        self.concat_dim = self.hidden_ques_dim + self.hidden_ans_dim + \
                          self.hidden_ans_dim + self.hidden_img_dim + \
                          self.hidden_cap_dim + self.hidden_hist_dim*9

        self.simnet = nn.Sequential(
            nn.Linear(self.concat_dim, (self.concat_dim)//2, bias=False),
            nn.BatchNorm1d((self.concat_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear((self.concat_dim)//2, (self.concat_dim)//4, bias=False),
            nn.BatchNorm1d((self.concat_dim) // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear((self.concat_dim)//4, 1)
        )

        # To share weights, provide list of tuples: (idx, list of connected utils)
        # Note, for efficiency, the shared utils (i.e., history, are connected to ans and question only.
        # connecting shared factors is not supported (!)
        sharing_factor_weights = {4: (9, [0, 1]),
                                  5: (9, [0, 1])}

        self.mul_atten = Atten(util_e=[self.hidden_ans_dim, # Answer modal
                                       self.hidden_ques_dim, # Question modal
                                       self.hidden_cap_dim, # Caption modal
                                       self.hidden_img_dim, # Image modal
                                       self.hidden_hist_dim, # Question-history modal
                                       self.hidden_hist_dim # Answer-history modal
                                       ],
                               sharing_factor_weights=sharing_factor_weights,
                               sizes=[100, # 100 Answers
                                      21, # Question length
                                      41, # Caption length
                                      37, # 36 Image regions
                                      21, # History-Question length
                                      21 #  History-Answer length
                                      ] # The spatial dim used for pairwise normalization (use force for adaptive)
                               , prior_flag=True,
                               pairwise_flag=True)



    def forward(self, input_ques, input_ans, input_hist_ques, input_hist_ans, input_hist_cap,
                input_ques_length, input_ans_length, input_cap_length, i_e):
        """

        :param input_ques:
        :param input_ans:
        :param input_hist_ques:
        :param input_hist_ans:
        :param input_hist_cap:
        :param input_ques_length:
        :param input_ans_length:
        :param input_cap_length:
        :param i_e:
        :return:
        """


        n_options = input_ans.size()[1]
        batch_size = input_ques.size()[0]



        nqa_per_dial, nwords_per_qa = input_hist_ques.size()[1], input_hist_ques.size()[2]
        nwords_per_cap = input_hist_cap.size()[1]
        max_length_input_ans = input_ans.size()[-1]

        assert batch_size == input_hist_ques.size()[0] == input_hist_ans.size()[0] == input_ques.size()[0] == \
               input_ans.size()[0] == input_hist_cap.size()[0]
        assert nqa_per_dial == input_hist_ques.size()[1] == input_hist_ans.size()[1]
        assert nwords_per_qa == input_hist_ques.size()[2] == input_hist_ans.size()[2]

        q_we = self.word_embedddings(input_ques)
        a_we = self.word_embedddings(input_ans.view(-1, max_length_input_ans))
        hq_we = self.word_embedddings(input_hist_ques.view(-1, nwords_per_qa))
        ha_we = self.word_embedddings(input_hist_ans.view(-1, nwords_per_qa))
        c_we = self.word_embedddings(input_hist_cap.view(-1, nwords_per_cap))



        '''
            q_we = batch x 20 x embed_ques_dim
            a_we = 100*batch x 20 x embed_ans_dim
            hq_we = batch*nqa_per_dial, nwords_per_qa, embed_hist_dim
            ha_we = batch*nqa_per_dial, nwords_per_qa, embed_hist_dim
            c_we = batch*ncap_per_dial, nwords_per_cap, embed_hist_dim
        '''
        self.lstm_ques.flatten_parameters()
        self.lstm_ans.flatten_parameters()
        self.lstm_hist_ques.flatten_parameters()
        self.lstm_hist_ans.flatten_parameters()
        self.lstm_hist_cap.flatten_parameters()


        i_feat = i_e

        q_seq, self.hidden_ques = self.lstm_ques(q_we)
        a_seq, self.hidden_ans = self.lstm_ans(a_we)
        hq_seq, self.hidden_hist_ques = self.lstm_hist_ques(hq_we)
        ha_seq, self.hidden_hist_ans = self.lstm_hist_ans(ha_we)
        cap_seq, self.hidden_cap = self.lstm_hist_cap(c_we)


        '''
            length is used for attention prior
        '''
        q_len = input_ques_length.data - 1
        c_len = input_cap_length.data.view(-1) - 1


        ans_index = torch.arange(0, n_options * batch_size).long().cuda()
        ans_len = input_ans_length.data.view(-1) - 1
        ans_seq = a_seq[ans_index, ans_len, :]
        ans_seq = ans_seq.view(batch_size, n_options, self.hidden_ans_dim)

        batch_index = torch.arange(0, batch_size).long().cuda()
        q_prior = torch.zeros(batch_size, q_seq.size(1)).cuda()
        q_prior[batch_index, q_len] = 100
        c_prior = torch.zeros(batch_size, cap_seq.size(1)).cuda()
        c_prior[batch_index, c_len] = 100
        ans_prior = torch.ones(batch_size, ans_seq.size(1)).cuda()
        img_prior = torch.ones(batch_size, i_feat.size(1)).cuda()

        (ans_atten, ques_atten, cap_atten, img_atten, hq_atten, ha_atten) = \
            self.mul_atten([ans_seq, q_seq, cap_seq, i_feat, hq_seq, ha_seq],
                           priors=[ans_prior, q_prior, c_prior, img_prior, None, None])

        '''
            expand to answers based
        '''
        ques_atten = torch.unsqueeze(ques_atten, 1).expand(batch_size,
                                                           n_options,
                                                           self.hidden_ques_dim)
        cap_atten = torch.unsqueeze(cap_atten, 1).expand(batch_size,
                                                         n_options,
                                                         self.hidden_cap_dim)
        img_atten = torch.unsqueeze(img_atten, 1).expand(batch_size, n_options,
                                                         self.hidden_img_dim)
        ans_atten = torch.unsqueeze(ans_atten, 1).expand(batch_size, n_options,
                                                         self.hidden_ans_dim)


        '''
            combine history
        '''

        input_qahistnet = torch.cat((hq_atten, ha_atten), 1)
        # input_qahistnet: (nqa_per_dial*batch x 2*hidden_hist_dim)
        output_qahistnet = self.qahistnet(input_qahistnet)
        # output_qahistnet: (nqa_per_dial*batch x hidden_hist_dim)
        output_qahistnet = output_qahistnet.view(batch_size,
                                                 nqa_per_dial * self.hidden_hist_dim)
        # output_qahistnet: (batch x nqa_per_dial*hidden_hist_dim)
        output_qahistnet = torch.unsqueeze(output_qahistnet, 1)\
            .expand(batch_size,
                    n_options,
                    nqa_per_dial * self.hidden_hist_dim)

        input_qa = torch.cat((ans_seq, ques_atten, ans_atten, img_atten,
                              output_qahistnet, cap_atten), 2)  # Concatenate last dimension

        input_qa = input_qa.view(batch_size * n_options, self.concat_dim)

        out_scores = self.simnet(input_qa)

        out_scores = out_scores.squeeze(dim=1)
        out_scores = out_scores.view(batch_size, n_options)

        return out_scores