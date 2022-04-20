# Transformer库 研究笔记
bert-base-cased可以处理大写词汇 bert-base-uncased只能处理小写

# bert-base-cased/uncased模型下的config.json文件为模型参数的 真实复制 更改config中vocab_size等参数无法真正更改模型架构 且只会报错
***
custom模型参数的方法：直接在原始config.json文件夹中修改部分参数，复制一份保存到新的位置，在from_pretrained函数中利用config指令调用
example：
//my_config = './bert-custom/config.json'
//tokenizer = BertForMaskedLM.from_pretrained(model, config=my_config)
***

# 将TAPE数据集文件夹 直接放在Sparks/data下即可
# 注意 secondary_structure 下的split数据集有 train valid casp12 ts115 cb513 其它任务按需更改
# 若要使用官方的tape命令测试 需保证transformer模型的架构部分 embeddings vocab_size相同
tape-eval transformer secondary_structure pretrained_models/valid --metrics accuracy --split train --num_workers 0 --batch_size 4

tape-train transformer secondary_structure --num_workers 0 --batch_size 1

