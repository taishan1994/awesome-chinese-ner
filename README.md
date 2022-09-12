# awesome-chinese-ner
中文命名实体识别

#### 延申
- 中文预训练模型综述 <br>
https://www.jsjkx.com/CN/10.11896/jsjkx.211200018
- 中文预训练模型下载地址<br>
https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models
- 中文词向量下载地址<br>
https://github.com/Embedding/Chinese-Word-Vectors

#### 命名实体识别综述（中文）
- 基于深度学习的中文命名实体识别最新研究进展综述<br>
2022年 中文信息学报<br>
http://61.175.198.136:8083/rwt/125/http/GEZC6MJZFZZUPLSSGM3B/Qikan/Article/Detail?id=7107633068<br>
- 命名实体识别方法研究综述<br>
2022年 计算机科学与探索<br>
http://fcst.ceaj.org/CN/10.3778/j.issn.1673-9418.2112109<br>
- 中文命名实体识别综述<br>
2021年 计算机科学与探索<br>
http://fcst.ceaj.org/CN/abstract/abstract2902.shtml<br>

# 模型
- Domain-Specific NER via Retrieving Correlated Samples<br>
COLING 2022<br>
https://arxiv.org/pdf/2208.12995.pdf<br>
- Robust Self-Augmentation for Named Entity Recognition with Meta Reweighting<br>
NAACL 2022 <br>
https://arxiv.org/pdf/2204.11406.pdf<br>
https://github.com/LindgeW/MetaAug4NER<br>
- Boundary Smoothing for Named Entity Recognition<br>
ACL 2022<br>
https://arxiv.org/pdf/2204.12031v1.pdf<br>
https://github.com/syuoni/eznlp<br>
- NFLAT: Non-Flat-Lattice Transformer for Chinese Named Entity Recognition <br>
2022 <br>
https://arxiv.org/pdf/2205.05832.pdf <br>
- Unified Structure Generation for Universal Information Extraction <br>
（一统实体识别、关系抽取、事件抽取、情感分析） <br>
ACL 2022 <br>
https://arxiv.org/pdf/2203.12277.pdf <br>
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie <br>
https://github.com/universal-ie/UIE <br>
以下这篇也是通用的，只是英文方面的，没有中文数据上的实验：
  - DEEPSTRUCT: Pretraining of Language Models for Structure Prediction <br>
  2022<br>
  https://arxiv.org/pdf/2205.10475v1.pdf<br>
  https://github.com/cgraywang/deepstruct<br>

- Parallel Instance Query Network for Named Entity Recognition <br>
2022 <br>
https://arxiv.org/pdf/2203.10545v1.pdf <br>
- Delving Deep into Regularity: A Simple but Effective Method for Chinese Named Entity Recognition<br>
NAACL 2022<br>
https://arxiv.org/pdf/2204.05544.pdf<br>
- TURNER: The Uncertainty-based Retrieval Framework for Chinese NER<br>
2022<br>
https://arxiv.org/pdf/2202.09022 <br>
- NN-NER: Named Entity Recognition with Nearest Neighbor Search<br>
2022 <br>
https://arxiv.org/pdf/2203.17103<br>
https://github.com/ShannonAI/KNN-NER<br>
- Unified Named Entity Recognition as Word-Word Relation Classification<br>
AAAI 2022 <br>
https://arxiv.org/abs/2112.10070 <br>
https://github.com/ljynlp/W2NER.git<br>
- MarkBERT: Marking Word Boundaries Improves Chinese BERT<br>
2022 <br>
https://arxiv.org/pdf/2203.06378<br>
- MFE-NER: Multi-feature Fusion Embedding for Chinese Named Entity Recognition<br>
2021 <br>
https://arxiv.org/pdf/2109.07877<br>
- AdaK-NER: An Adaptive Top-K Approach for Named Entity Recognition with Incomplete Annotations<br>
2021 <br>
https://arxiv.org/pdf/2109.05233<br>
- ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information <br>
ACL 2021<br>
https://arxiv.org/pdf/2106.16038<br>
https://github.com/ShannonAI/ChineseBert<br>
- Enhanced Language Representation with Label Knowledge for Span Extraction<br>
EMNLP 2021<br>
https://aclanthology.org/2021.emnlp-main.379.pdf<br>
https://github.com/Akeepers/LEAR <br>
- Lex-BERT: Enhancing BERT based NER with lexicons  <br>
ICLR 2021 <br>
https://arxiv.org/pdf/2101.00396v1.pdf<br>
- Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter <br>
ACL 2021 <br>
https://arxiv.org/pdf/2105.07148.pdf<br>
https://github.com/liuwei1206/LEBERT<br>
- MECT: Multi-Metadata Embedding based Cross-Transformer for Chinese Named Entity Recognition  <br>
ACL 2021  <br>
https://arxiv.org/pdf/2107.05418v1.pdf<br>
https://github.com/CoderMusou/MECT4CNER <br>
- Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition <br>
ACL 2021   <br>
https://arxiv.org/pdf/2105.06804v2.pdf<br>
https://github.com/tricktreat/locate-and-label <br>
- Dynamic Modeling Cross- and Self-Lattice Attention Network for Chinese NER<br>
AAAI 2021<br>
https://ojs.aaai.org/index.php/AAAI/article/view/17706/17513<br>
https://github.com/zs50910/DCSAN-for-Chinese-NER<br>
- Improving Named Entity Recognition with Attentive Ensemble of Syntactic Information  <br>
EMNLP-2020 <br>
https://arxiv.org/pdf/2010.15466  <br>
https://github.com/cuhksz-nlp/AESINER <br>
- ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations  <br>
ACL 2020  <br>
https://arxiv.org/pdf/1911.00720v1.pdf<br>
https://github.com/sinovation/ZEN <br>
- A Unified MRC Framework for Named Entity Recognition  <br>
ACL 2020  <br>
https://arxiv.org/pdf/1910.11476v6.pdf<br>
https://github.com/ShannonAI/mrc-for-flat-nested-ner <br>
- Simplify the Usage of Lexicon in Chinese NER  <br>
ACL 2020   <br>
https://arxiv.org/pdf/1908.05969.pdf<br>
https://github.com/v-mipeng/LexiconAugmentedNER <br>
- Dice Loss for Data-imbalanced NLP Tasks  <br>
ACL 2020  <br>
https://arxiv.org/pdf/1911.02855v3.pdf<br>
https://github.com/ShannonAI/dice_loss_for_NLP <br>
- Porous Lattice Transformer Encoder for Chinese NER<br>
COLING 2020<br>
https://aclanthology.org/2020.coling-main.340.pdf<br>
- FLAT: Chinese NER Using Flat-Lattice Transformer  <br>
ACL 2020  <br>
https://arxiv.org/pdf/2004.11795v2.pdf<br>
https://github.com/LeeSureman/Flat-Lattice-Transformer <br>
- FGN: Fusion Glyph Network for Chinese Named Entity Recognition  <br>
2020  <br>
https://arxiv.org/pdf/2001.05272v6.pdf  <br>
https://github.com/AidenHuen/FGN-NER<br>
- SLK-NER: Exploiting Second-order Lexicon Knowledge for Chinese NER <br>
2020 <br>
https://arxiv.org/pdf/2007.08416v1.pdf<br>
https://github.com/zerohd4869/SLK-NER <br>
- Entity Enhanced BERT Pre-training for Chinese NER<br>
EMNLP 2020<br>
https://aclanthology.org/2020.emnlp-main.518.pdf<br>
https://github.com/jiachenwestlake/Entity_BERT<br>
- Improving Named Entity Recognition with Attentive Ensemble of Syntactic Information  <br>
ACL2020 <br>
https://arxiv.org/pdf/2010.15466v1.pdf<br>
https://github.com/cuhksz-nlp/AESINER  <br>
- Named Entity Recognition for Social Media Texts with Semantic Augmentation  <br>
EMNLP 2020  <br>
https://arxiv.org/pdf/2010.15458v1.pdf<br>
https://github.com/cuhksz-nlp/SANER <br>
- CLUENER2020: Fine-grained Named Entity Recognition Dataset and Benchmark for Chinese  <br>
2020  <br>
https://arxiv.org/pdf/2001.04351v4.pdf<br>
https://github.com/CLUEbenchmark/CLUENER2020 <br>
- ERNIE: Enhanced Representation through Knowledge Integration  <br>
2019  <br>
https://arxiv.org/pdf/1904.09223v1.pdf<br>
https://github.com/PaddlePaddle/ERNIE <br>
- TENER: Adapting Transformer Encoder for Named Entity Recognition  <br>
2019  <br>
https://arxiv.org/pdf/1911.04474v3.pdf<br>
https://github.com/fastnlp/TENER <br>
- Chinese NER Using Lattice LSTM  <br>
ACL 2018  <br>
https://arxiv.org/pdf/1805.02023v4.pdf<br>
https://github.com/jiesutd/LatticeLSTM <br>
- ERNIE 2.0: A Continual Pre-training Framework for Language Understanding  <br>
2019  <br>
https://arxiv.org/pdf/1907.12412v2.pdf<br>
https://github.com/PaddlePaddle/ERNIE <br>
- Glyce: Glyph-vectors for Chinese Character Representations  <br>
NeurIPS 2019  <br>
https://arxiv.org/pdf/1901.10125v5.pdf<br>
https://github.com/ShannonAI/glyce <br>
- CAN-NER: Convolutional Attention Network for Chinese Named Entity Recognition   <br>
NAACL 2019   <br>
https://arxiv.org/pdf/1904.02141v3.pdf<br>
https://github.com/microsoft/vert-papers/tree/master/papers/CAN-NER<br>
- Neural Chinese Named Entity Recognition via CNN-LSTM-CRF and Joint Training with Word Segmentation  <br>
2019  <br>
https://arxiv.org/pdf/1905.01964v1.pdf<br>
https://github.com/rxy007/cnn-lstm-crf <br>
- Chinese Named Entity Recognition Augmented with Lexicon Memory  <br>
2019  <br>
https://arxiv.org/pdf/1912.08282v2.pdf<br>
https://github.com/dugu9sword/LEMON <br>
- Exploiting Multiple Embeddings for Chinese Named Entity Recognition   <br>
2019  <br>
https://arxiv.org/pdf/1908.10657v1.pdf<br>
https://github.com/WHUIR/ME-CNER <br>
- Dependency-Guided LSTM-CRF for Named Entity Recognition <br>
IJCNLP 2019  <br>
https://arxiv.org/pdf/1909.10148v1.pdf<br>
https://github.com/allanj/ner_with_dependency <br>
- CAN-NER: Convolutional Attention Network for Chinese Named Entity Recognition <br>
NAACL-HLT (1) 2019 <br>
https://aclanthology.org/N19-1342/ <br>
- CNN-Based Chinese NER with Lexicon Rethinking <br>
IJCAI 2019 <br>
https://www.ijcai.org/proceedings/2019/0692.pdf <br>
https://aclanthology.org/N19-1342.pdf <br>
- Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network <br>
IJCNLP 2019  <br>
https://aclanthology.org/D19-1396.pdf<br>
https://github.com/DianboWork/Graph4CNER <br>
- Distantly Supervised NER with Partial Annotation Learning and Reinforcement Learning  <br>
COLING 2018   <br>
https://aclanthology.org/C18-1183.pdf<br>
https://github.com/rainarch/DSNER <br>
- Adversarial Transfer Learning for Chinese Named Entity Recognition with Self-Attention Mechanism  <br>
EMNLP 2018  <br>
https://aclanthology.org/D18-1017.pdf<br>
https://github.com/CPF-NLPR/AT4ChineseNER <br>

# 非中文模型
没有针对于中文的实验，但是思想可以借鉴的： <br>
- A Unified Generative Framework for Various NER Subtasks <br>
（使用BART生成模型进行命名实体识别） <br>
ACL-ICJNLP 2021 <br>
https://arxiv.org/pdf/2106.01223.pdf <br>
https://github.com/yhcc/BARTNER <br>
(以下四篇是基于prompt的命名实体识别) <br>
- Template-Based Named Entity Recognition Using BART <br>
https://arxiv.org/abs/2106.01760 <br>
https://github.com/Nealcly/templateNER <br>
- Good Examples Make A Faster Learner: Simple Demonstration-based Learning for Low-resource NER <br>
https://arxiv.org/abs/2110.08454 <br>
https://github.com/INK-USC/fewNER <br>
- LightNER: A Lightweight Generative Framework with Prompt-guided Attention for Low-resource NER <br>
https://arxiv.org/abs/2109.00720 <br>
https://github.com/zjunlp/DeepKE/blob/main/example/ner/few-shot/README_CN.md <br>
- Template-free Prompt Tuning for Few-shot NER <br>
https://arxiv.org/abs/2109.13532 <br>
https://github.com/rtmaww/EntLM/ <br>


# 数据集

- [MSRA](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/MSRA)
- [Weibo](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/weibo)
- [resume](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/ResumeNER )
- onenotes4
- onenotes5
- [一家公司提供的数据集,包含人名、地名、机构名、专有名词。](https://bosonnlp.com/dev/resource)
- [人民网（04年）](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/people_daily)
- [影视-音乐-书籍实体标注数据](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/video_music_book_datasets)
- [中文医学文本命名实体识别 2020CCKS](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/2020_ccks_ner)
- [医渡云实体识别数据集](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/yidu-s4k )
- [CLUENER2020](https://github.com/GuocaiL/nlp_corpus/tree/main/open_ner_data/cluener_public)
- [不同任务中文数据集整理](https://github.com/liucongg/NLPDataSet)

# 预训练语言模型

- ChineseBert
- MacBert
- SpanBert
- XLNet
- Roberta
- Bert
- StructBert
- WoBert
- ELECTRA
- Ernie1.0
- Ernie2.0
- Ernie3.0
- NeZha
- MengZi
- [Pretraining without Wordpieces: Learning Over a Vocabulary of Millions of Words](https://arxiv.org/pdf/2202.12142)
- [PERT: Pre-Training BERT with Permuted Language Model](https://arxiv.org/abs/2203.06906)

# Ner工具

- [Stanza](https://github.com/stanfordnlp/stanza)
- [LAC](https://github.com/baidu/lac)
- [Ltp](https://github.com/HIT-SCIR/ltp)
- [Hanlp](https://github.com/hankcs/HanLP)
- [foolnltk](https://github.com/rockyzhengwu/FoolNLTK)
- [NLTK](https://github.com/nltk/nltk)
- BosonNLP
- [FudanNlp](https://github.com/FudanNLP/fnlp)
- [Jionlp](https://github.com/dongrixinyu/JioNLP)
- [HarvestText](https://github.com/blmoistawinde/HarvestText)
- [fastHan](https://github.com/fastnlp/fastHan)

# 比赛

- CCKS2017开放的中文的电子病例测评相关的数据。<br>
评测任务一：https://biendata.com/competition/CCKS2017_1/<br>
评测任务二：https://biendata.com/competition/CCKS2017_2/<br>
- CCKS2018开放的音乐领域的实体识别任务。<br>
评测任务：https://biendata.com/competition/CCKS2018_2/<br>
- (CoNLL 2002)Annotated Corpus for Named Entity Recognition。<br>
地址：https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus<br>
- NLPCC2018开放的任务型对话系统中的口语理解评测。<br>
地址：http://tcci.ccf.org.cn/conference/2018/taskdata.php<br>
- 非结构化商业文本信息中隐私信息识别<br>
地址：https://www.datafountain.cn/competitions/472/datasets
- 商品标题识别<br>
地址：https://www.heywhale.com/home/competition/620b34ed28270b0017b823ad/content/3
- CCKS2021中文NLP地址要素解析<br>
地址：https://tianchi.aliyun.com/competition/entrance/531900/introduction
