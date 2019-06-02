import tensorflow as tf
class Config:
    #训练参数
    optim = 'adam'
    learning_rate = 0.001
    weight_decay = 0
    dropout_keep_prob = 1
    batch_size = 32
    epochs = 10
    #模型设置
    algo = 'BIDAF'
    embed_size = 300
    hidden_size = 150
    max_p_num = 5#在一个simple中，最大的passage数
    max_p_len = 500#最大文章长度，分词后
    max_q_len = 60#最大问题长度，分词后
    max_a_len = 200#最大答案长度
    #预训练词向量
    prepared_dir = '../data/demo/pre_embeddings/'
    #数据集参数
    train_files = ['../data/demo/trainset/search.train.json']
    dev_files = ['../data/demo/devset/search.dev.json']
    test_files = ['../data/demo/testset/search.test.json']
    #模型保存路径
    brc_dir = '../data/demo/baidu'
    vocab_dir = '../data/demo/vocab/'
    model_dir = '../data/demo/models/'
    result_dir = '../data/demo/results/'
    summary_dir = '../data/demo/summary/'
    log_dir = ''