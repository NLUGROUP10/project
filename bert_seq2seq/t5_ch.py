
import torch
from bert_seq2seq.model.t5_model import T5ForConditionalGeneration, T5Config, T5SmallConfig
from bert_seq2seq.tokenizer import T5PegasusTokenizer,load_chinese_base_vocab
from bert_seq2seq.basic_bert import BasicT5
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
import torch.nn.functional as F


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty  # 长度惩罚的指数系数
        self.num_beams = num_beams  # beam size
        self.beams = []  # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9  # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty  # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
            # 如果类没装满num_beams个序列
            # 或者装满以后，但是待加入序列的score值大于类中的最小值
            # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
        # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
        # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class T5Model(BasicT5):

    def __init__(self, word2idx, size="base"):
        super().__init__()
        if size == "base":
            config = T5Config()
        elif size == "small":
            config = T5SmallConfig()
        else:
            raise Exception("not support this model type")
        self.model = T5ForConditionalGeneration(config)

        self.word2idx = word2idx
        self.tokenizer = T5PegasusTokenizer(self.word2idx)
        self.bos_id = self.word2idx["[CLS]"]
        self.eos_id = self.word2idx["[SEP]"]
        self.unk_id = self.word2idx["[UNK]"]
        self.pad_id = self.word2idx['[PAD]']

    def forward(self, input_ids, decoder_input_ids, labels=None):
        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)


    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=True):

        token_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(token_out) == 2:
            token_ids = token_out[0]
        else:
            token_ids = token_out
        if not add_eos:
            token_ids = token_ids[:-1]
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []

        input_decoder_ids = torch.tensor(self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.model(input_ids=token_ids, decoder_input_ids=input_decoder_ids)[0]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.unk_id] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if self.eos_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                input_decoder_ids = torch.cat((input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids)


    def beam_search(self, input_text,beam_size,input_max_length = 256,out_max_length = 50):
        self.out_max_length = out_max_length
        input_token_ids = [input_text for _ in range(beam_size)]
        input_token_ids = self.tokenizer.batch_encode(input_token_ids,self.pad_id,max_length=input_max_length)
        input_token_ids = torch.tensor(input_token_ids,dtype=torch.long,device=self.device)

        input_decoder_ids = torch.tensor(self.bos_id, device=self.device, dtype=torch.long).repeat(beam_size,1).view(beam_size, -1)
        beam_scores = torch.zeros((1, beam_size),device=self.device)
        beam_scores[:, 1:] = -1e9
        complete_seqs = []
        complete_seqs_scores = []
        with torch.no_grad():
            # output_scores = torch.zeros(input_token_ids.shape[0],device=self.device)
            for step in range(self.out_max_length):
                outputs = self.model.forward(input_ids=input_token_ids,decoder_input_ids=input_decoder_ids)[0]
                next_token_logits = outputs[:, -1, :]
                scores = F.log_softmax(next_token_logits, dim=-1) #[beam,voc]
                vocab_size = scores.size(-1)
                next_scores = scores + beam_scores.view(beam_size,-1).expand_as(scores)
                next_scores = next_scores.view(
                    1, beam_size*vocab_size
                )
                next_scores, next_tokens = torch.topk(next_scores, beam_size, dim=1, largest=True, sorted=True)
                beam_id = next_tokens // vocab_size  # 行索引
                new_token_id = next_tokens % vocab_size

                beam_scores = next_scores

                input_decoder_ids = torch.cat([input_decoder_ids[beam_id].squeeze(0),new_token_id.view(beam_size,-1)],dim=-1)

                end_counts = (input_decoder_ids == self.eos_id).sum(1)  # 统计出现的end标记
                best_one = beam_scores.argmax()
                if end_counts[best_one] == 1:
                    return input_decoder_ids[best_one][:-1]
                else:
                    flag = (end_counts < 1)
                    if not flag.all():
                        input_token_ids = input_token_ids[flag]
                        input_decoder_ids = input_decoder_ids[flag]
                        beam_scores = beam_scores.view(beam_size,-1)[flag].view(1,-1)
                        beam_size = flag.sum()
            return input_decoder_ids[beam_scores.argmax()]




















    def batched_beam_search(self, src_tokens, num_beams, memory,max_in_length = 256, max_out_length=200, pad_token_id=0, sos_token_id=1, eos_token_id=2,
                    length_penalty=1):
        # //todo finish batch encodiing
        decoder = None

        token_out = self.tokenizer.encode(src_tokens,max_in_length)


        batch_size, box_num, d_model = memory.size() #[b,len,hidden]
        beam_scores = torch.zeros((batch_size, num_beams)).to('cuda')  # 定义scores向量，保存累加的log_probs #[b,beam]
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf [[0,-10000],[0,-10000]]
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams),1
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(num_beams, max_out_length, length_penalty=length_penalty)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long).to('cuda') #[b*beam,1]
        memory = memory.repeat(1, num_beams, 1).reshape(batch_size * num_beams, box_num, d_model) #[b*beam,len,hidden]
        cur_len = 1
        while cur_len < max_out_length:
            # outputs: (batch_size*num_beams, cur_len, vocab_size)
            outputs = decoder(memory, input_ids)
            # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
            next_token_logits = outputs[:, -1, :]# (batch_size*num_beams, vocab_size)
            scores = F.log_softmax(next_token_logits, dim=-1)  # log_softmax # (batch_size*num_beams, vocab_size)
            vocab_size = scores.size(-1) # (batch_size*num_beams* vocab_size)
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # 累加上以前的scores [(batch_size * num_beams),vocab_size] = [(batch_size * num_beams),vocab_size]+[(batch_size * num_beams),vocab_size]
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size #[batch_size , num_beams*vocab_size]
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // vocab_size  # 1
                    token_id = beam_token_id % vocab_size  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * num_beams + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == num_beams:
                        break
                    # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
            # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            # 准备下一次循环(下一层的解码)
            # beam_scores: (num_beams * batch_size)
            # beam_tokens: (num_beams * batch_size)
            # beam_idx: (num_beams * batch_size)
            # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
            # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(num_beams):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, max_out_length)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_out_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded.to('cpu')

    def greedy(model, data, vocab):
        '''贪婪策略解码'''
        bs = len(data['img_ids'])
        data['feats'] = data['feats']
        fixed_len = model.config.fixed_len
        inp = torch.ones(bs, 1).to('cuda').long()
        data['text_masks'] = None
        for step in range(fixed_len + 1):
            data['texts'] = inp
            outputs = model(data)
            score, token_id = torch.max(outputs['logits'][:, -1, :].squeeze(1), dim=-1)
            inp = torch.cat([inp, token_id.unsqueeze(1)], dim=1)

        sent = [vocab.idList_to_sent(idList) for idList in inp.to('cpu')]
        return sent


if __name__ == '__main__':
    # decodetest
    train_data_path = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\test.ape.json"
    val_data_path = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\test.ape.json"

    vocab_path = r"D:\codeproject\NLP\models\chinese_t5_pegasus_small\vocab.txt"
    model_path = r"D:\codeproject\NLP\models\chinese_t5_pegasus_small\pytorch_model.bin"
    model_save_path = r"D:\codeproject\NLP\models\chinese_t5_pegasus_small\t5_ancient_trans_model.bin"
    batch_size = 8
    lr = 1e-5
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenizer = T5PegasusTokenizer(word2idx)
    model = T5Model(word2idx, size="small")
    model.load_pretrain_params(model_path)
    tst = "王艳家买了一台洗衣机和一台电冰箱，一共花了6000元，电冰箱的价钱是洗衣机的3/5，求洗衣机的价钱．"
    res = model.beam_search(tst,beam_size=5)
    print(res)
    print(tokenizer.decode(res.tolist()))
    res2 = model.sample_generate_encoder_decoder(tst)
    print(res2)