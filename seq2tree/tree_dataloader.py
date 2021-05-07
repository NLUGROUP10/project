import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from bert_seq2seq.tokenizer import load_chinese_base_vocab, T5PegasusTokenizer

val_str = '["95", "##90", "154", "38", "395", "000", "790", "6", "245", "394", "06", "15000", "680", "1908", "178", "148", "0", "226", "055", "1800", "161", "2300", "139", "##87", "1949", "265", "122", "-", "900", "372", "1936", "590", "705", "302", "13", ".", "02", "347", "399", "193", "365", "508", "183", "3800", "213", "455", "290", "159", "##83", "328", "304", "149", "##2007", "181", "1963", "##34", "393", "203", "1978", "337", "468", "5000", "246", "03", "209", "1968", "426", "207", "802", "##009", "78", "##000", "##520", "428", "580", "##38", "440", "025", "256", "252", "2006", "##12", "422", "208", "100", "10000", "661", "##22", "##999", "6500", "313", "980", "3200", "310", "##2008", "86", "##47", "402", "1993", "60", "392", "##110", "390", "2008", "264", "24", "##1000", "##85", "5500", "1300", "2013", "102", "101", "##16", "##46", "182", "389", "588", "1998", "##2012", "515", "81", "214", "7000", "110", "##78", "194", "9000", "1001", "93", "322", "47", "344", "##49", "237", "073", "377", "2002", "1860", "022", "1600", "414", "##82", "empty", "##010", "830", "450", "##53", "530", "4000", "##2009", "##120", "413", "241", "68", "610", "371", "61", "230", "1946", "118", "1985", "1", "21", "170", "89", "##43", "361", "1974", "258", "411", "243", "620", "560", "835", "##2014", "1933", "289", "1907", "407", "1973", "458", "##365", "1931", "270", "445", "92", "412", "768", "220", "460", "1986", "##27", "403", "525", "126", "29", "315", "166", "2050", "330", "##50", "334", "##67", "197", "##09", "1900", "840", "##06", "70", "348", "540", "405", "1400", "15", "##24", "43", "##91", "##5", "075", "320", "1010", "1947", "364", "132", "1990", "1890", "005", "20", "##360", "1964", "##2015", "12", "157", "120", "295", "8000", "218", "##33", "49", "355", "18", "539", "366", "1935", "##65", "2500", "1920", "##2013", "4200", "425", "##001", "107", "42", "50", "66", "48", "318", "162", "770", "##2000", "520", "2007", "020", "267", "138", "63", "113", "1972", "##75", "31", "##2", "##900", "##52", "121", "216", "##7", "1989", "##e", "77", "##28", "1897", "1896", "370", "1945", "##77", "103", "818", "##76", "151", "150", "555", "##68", "720", "##61", "504", "1937", "441", "505", "177", "18000", "308", "010", "##470", "013", "##00", "30000", "416", "323", "486", "1976", "##60", "1930", "238", "##41", "485", "234", "1898", "921", "##8", "406", "513", "##19", "998", "41", "616", "##40", "12345", "##42", "378", "276", "2000", "10", "143", "59", "415", "224", "1888", "660", "385", "640", "83", "##500", "978", "7500", "2022", "17", "90", "475", "201", "311", "925", "316", "501", "##888", "123456", "777", "1921", "117", "386", "383", "16", "2018", "204", "1992", "07", "274", "281", "147", "44", "1885", "510", "1500", "357", "119", "1924", "409", "206", "106", "618", "52", "300", "##17", "327", "1280", "105", "1982", "480", "368", "##580", "2030", "1024", "312", "1234", "##79", "2700", "25000", "369", "33", "202", "229", "301", "503", "211", "9", "1932", "1965", "6000", "##9", "##14", "244", "##64", "456", "329", "388", "701", "1080", "349", "999", "860", "1950", "536", "80", "173", "35", "345", "798", "73", "##80", "54", "125", "280", "870", "163", "69", "##007", "1910", "1966", "251", "550", "314", "930", "2010", "191", "05", "023", "522", "801", "391", "511", "144", "1000", "##18", "160", "421", "36", "##92", "430", "298", "96", "247", "1971", "650", "029", "1997", "1980", "321", "##23", "##73", "332", "##25", "499", "852", "2016", "451", "488", "324", "360", "1914", "453", "12000", "##600", "838", "346", "730", "##200", "272", "221", "384", "396", "25", "278", "187", "367", "326", "##72", "/", "275", "##96", "##37", "1977", "780", "299", "219", "028", "##3000", "2020", "688", "886", "710", "3400", "+", "45", "##800", "189", "427", "=", "23", "1850", "##2010", "1200", "167", "261", "1967", "317", "x", "352", "850", "398", "1948", "2025", "##44", "1870", "155", "53", "2017", "26", "37", "205", "174", "27", "192", "307", "570", "2021", "179", "##6", "254", "32", "287", "1901", "4", "##1", "171", "##35", "1925", "##66", "##10", "253", "888", "158", "249", "##70", "4500", "293", "1840", "4800", "342", "666", "404", "19", "91", "##2011", "268", "1940", "124", "153", "1919", "990", "518", "690", "164", "356", "180", "225", "760", "880", "##700", "920", "98", "1938", "429", "350", "625", "595", "##88", "442", "1994", "1020", "222", "##21", "012", "1953", "709", "67", "309", "015", "##86", "338", "495", "297", "215", "112", "##02", "400", "816", "521", "50000", "740", "1999", "432", "227", "277", "1951", "58", "444", "262", "1960", "2011", "##750", "137", "228", "##30", "700", "##100", "543", "335", "1350", "259", "2015", "88", "354", "129", "##57", "502", "340", "457", "599", "2900", "##13", "##300", "024", "141", "##03", "##59", "##71", "34", "2009", "236", "3600", "006", "##08", "99", "176", "417", "87", "433", "195", "##05", "##3", "283", "145", "5", "1996", "353", "001", "319", "363", "373", "04", "303", "899", "104", "603", "212", "2600", "387", "190", "##11", "##97", "269", "008", "##74", "133", "135", "418", "004", "333", "1927", "109", "196", "123", "##56", "379", "1984", "240", "1944", "470", "199", "397", "1884", "273", "217", "56", "##123", "003", "1916", "186", "##4", "260", "##81", "380", "250", "30", "820", "292", "867", "##93", "1981", "002", "438", "296", "3500", "626", "116", "282", "8", "2001", "1880", "1979", "65", "448", "##95", "288", "374", "62", "799", "1988", "401", "2004", "498", "435", "2012", "248", "2005", "##62", "339", "3300", "01", "271", "2003", "184", "##84", "1905", "175", "##32", "285", "670", "4600", "72", "242", "##55", "3000", "007", "500", "2100", "970", "##36", "866", "985", "600", "223", "233", "##01", "257", "114", "200", "1991", "134", "111", "76", "284", "420", "##89", "506", "810", "1923", "108", "##250", "1955", "##39", "1956", "**", "279", "##94", "1983", "890", "##0", "009", "71", "286", "##58", "341", "306", "516", "239", "##48", "1700", "408", "##63", "131", "55", "419", "305", "210", "1911", "82", "828", "787", "800", "351", "294", "897", "061", "950", "165", "291", "263", "855", "2014", "863", "232", "601", "7", "343", "1962", "325", "382", "235", "362", "74", "1929", "##98", "376", "231", "##400", "1995", "1975", "130", "127", "##26", "1111", "512", "331", "359", "##99", "22", "266", "960", "79", "011", "336", "57", "##54", "146", "08", "28", "115", "255", "##07", "40", "019", "152", "##20", "1970", "465", "46", "97", "169", "39", "1987", "156", "94", "185", "2200", "358", "198", "51", "11", "##29", "64", "142", "##04", "14", "410", "699", "##31", "136", "75", "20000", "2400", "1250", "3", "128", "591", "611", "##013", "172", "381", "##69", "85", "##45", "168", "2345", "711", "*", "140", "1100", "2800", "84", "09", "3100", "375", "188", "814", "750", "630", "##101", "##15", "##51", "490", "2"]'


def generate_idx2word(vocab_list):
    idx2word = {i: w for i, w in enumerate(vocab_list)}
    return idx2word


def generate_word2idx(vocab_list):
    word2idx = {w: i for i, w in enumerate(vocab_list)}
    return word2idx


class Vocab:
    def __init__(self, vocab_list: list):
        self.list = []
        self.vocab_list = vocab_list
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.empty_token = '<empty>'
        self.vocab_list.append(self.empty_token)
        self.list.append(self.pad_token)
        self.list.append(self.unk_token)
        self.list.extend(self.vocab_list)
        # self.list.extend()
        self.word2idx = generate_word2idx(self.list)
        self.idx2word = generate_idx2word(self.list)

    def encode(self, input: List[str], max_len=512):
        input = input[:max_len]
        res = []
        for token in input:
            if token in self.vocab_list:
                res.append(self.word2idx[token])
            else:
                res.append(self.unk_token)
        return res

    def encode_rel(self, rels, max_len=512):
        # only for rels
        res = []
        rels = rels[:max_len]
        for line in rels:
            line = line[:max_len]
            idices = self.encode(line)
            res.append(idices)
        return res

    def decode(self, input: List[int]):
        return [self.idx2word[i] for i in input]

    def decode_rels(self, input: List[List[int]]):
        res = []
        for line in input:
            line = self.decode(line)
            res.append(line)
        return res


class Seq2TreeData(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, datapath, src_word2dix, values_vocab, types_vocab, max_depth=64):
        ## 一般init函数是加载所有数据
        super(Seq2TreeData, self).__init__()
        self.src_tokenizer = T5PegasusTokenizer(src_word2dix)
        self.values_vocab = Vocab(values_vocab)
        self.types_vocab = Vocab(types_vocab)
        self.depth = max_depth
        self.generate_rel_words()
        self.data = []
        with open(datapath, "r", encoding="utf-8") as file:
            for line in file:
                line = json.loads(line)
                line[-1] = json.loads(line[-1])
                self.data.append(line)
        print("data size is " + str(len(self.data)))

    def generate_rel_words(self):
        vocab = []
        for i in range(self.depth + 1):
            for j in range(self.depth + 1):
                word = f"{i}|{j}"
                vocab.append(word)
        self.rel_vocab = Vocab(vocab)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)

        single_data = self.data[i]
        text = single_data[0]
        dict = single_data[-1]
        token_ids_src, _ = self.src_tokenizer.encode(text, max_length=256)
        types_idx = self.types_vocab.encode(dict['types'])
        values_idx = self.values_vocab.encode(dict['values'])
        rels_idx = self.rel_vocab.encode_rel(dict['rels'])

        tgt_dict = {
            "types": types_idx,
            'values': values_idx,
            'paths': dict["paths"],
            'rels': rels_idx
        }
        output = {"text_idices": token_ids_src,
                  "target": tgt_dict}
        return output

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collect_funtion(batch: List[dict]):
        # src [b,maxlen]
        # values and types [b,maxlen]
        # positions [b,max_len, depth*width]
        # rels [b,maxlen,maxlen]

        def padding(indice, max_length, pad_idx=0):
            """
            pad 函数
            """
            pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
            res = torch.tensor(pad_indice)
            return res

        def padding_position(indices: List[List[List[int]]], max_len, wxd):
            # to_shape[max_len,wxd]
            padded_positions = [item + [[0 for _ in range(wxd)] for _ in range(max(0, max_len - len(item)))] for item in
                                indices]
            return torch.tensor(padded_positions)

        def padding_rels(rels, max_len, pad_idx=0):
            res = torch.zeros([len(rels), max_len, max_len], dtype=torch.int32)
            for i, item in enumerate(rels):
                nums = len(item)
                padded = pad_idx * torch.ones([max_len, max_len], dtype=torch.int)
                tensor = torch.tensor(item, dtype=torch.int)
                padded[:nums, :nums] = tensor
                res[i, :, :] = padded
            return res

        token_ids_src = [data["text_idices"] for data in batch]
        max_length_src = max([len(t) for t in token_ids_src])
        types_ids_tgt = [data["target"]['types'] for data in batch]
        max_length_tgt = max([len(t) for t in types_ids_tgt])
        values_ids_tgt = [data["target"]['values'] for data in batch]
        positions_ids_tgt = [data["target"]['paths'] for data in batch]
        rels_ids_tgt = [data["target"]['rels'] for data in batch]

        token_ids_padded = padding(token_ids_src, max_length_src)
        types_ids_padded = padding(types_ids_tgt, max_length_tgt)
        values_ids_padded = padding(values_ids_tgt, max_length_tgt)
        positions_ids_padded = padding_position(positions_ids_tgt, max_length_tgt,
                                                len(positions_ids_tgt[0][0]))  # [b,max_len,wxd]
        rels_ids_padded = padding_rels(rels_ids_tgt, max_length_tgt)  # [b max_len, max_len]

        labels_types_ids = types_ids_padded.clone()
        labels_values_ids = values_ids_padded.clone()
        labels_types_ids[labels_types_ids == 0] = -100
        labels_values_ids[labels_values_ids == 0] = -100

        values_ids_padded = values_ids_padded[:, :-1].contiguous()  # [b,maxlen-1]
        types_ids_padded = types_ids_padded[:, :-1].contiguous()  # [b,maxlen-1]
        positions_ids_padded = positions_ids_padded[:, :-1, :].contiguous()
        rels_ids_padded = rels_ids_padded[:, :-1, :-1].contiguous()
        labels_types_ids = labels_types_ids[:, 1:].contiguous()
        labels_values_ids = labels_values_ids[:, 1:].contiguous()

        return token_ids_padded, types_ids_padded, values_ids_padded, positions_ids_padded, rels_ids_padded, labels_types_ids, labels_values_ids


if __name__ == '__main__':
    vocab_path = r"D:\codeproject\NLP\models\chinese_t5_pegasus_small\vocab.txt"
    types = ['end_0',
             'float_1',
             'int_0',
             'int_1',
             'operator_1',
             'operator_2',
             'start_2',
             'subnum_0',
             'subnum_1',
             'var_0']
    values = json.loads(val_str)
    word2idx = load_chinese_base_vocab(vocab_path)
    outpath = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\cleaned\\test.ape.json"
    test_data = Seq2TreeData(outpath, word2idx, values, types)
    res = test_data.__getitem__(1)
    print(res)
    print(test_data.types_vocab.decode(res["target"]['types']))
    # print(test_data.values_vocab.decode(res['values']))
    # print(test_data.rel_vocab.decode_rels(res['rels']))
    dataloader = DataLoader(test_data,batch_size=1,shuffle=True,collate_fn=test_data.collect_funtion)

    for i,batch in enumerate(dataloader):
        if i==0:
            break
    print([item.shape for item in batch])