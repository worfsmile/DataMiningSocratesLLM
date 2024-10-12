import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorWithPadding
import datasets
import torch.nn as nn
import os

# 禁用CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有CUDA设备
# 设置设备为 CPU
device = torch.device("cpu")

# 加载预训练模型和tokenizer
# model_name = "facebook/blenderbot-400M-distill"
model_name = "google/flan-t5-base"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits

        # 从输入中提取标签
        labels = inputs['labels']

        # 计算损失
        # 使用交叉熵损失作为示例
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略标签为-100的项
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def inference(prompt):
    # 处理输入
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)  # 移动到模型所在设备
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)  # 创建 attention_mask

    # 生成答案
    generated_tokens_with_prompt = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024)

    # 解码生成的tokens
    generated_answer = tokenizer.decode(generated_tokens_with_prompt[0], skip_special_tokens=True)
    return generated_answer

def tokenize_function(examples):
    """
    Tokenize the examples.
    """
    text = [q + a for q, a in zip(examples['question'], examples['answers'])]
    tokenized_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    
    # 将tokenized_inputs['input_ids']作为labels列
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].clone()  # 克隆输入作为标签
    return tokenized_inputs.to(device)

def load_data():
    # 加载数据集
    fine_tune_data = datasets.load_dataset('json', data_files='data_translated.json')['train'].select([0, 10])
    tokenized_dataset = fine_tune_data.map(tokenize_function, batched=True, batch_size=1, drop_last_batch=True)

    # 选择保留的列
    columns_to_keep = ['input_ids', 'attention_mask', 'labels']
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names if col not in columns_to_keep])

    # 拆分数据集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    return train_dataset, test_dataset

def train_model():
    # 加载数据集
    train_dataset, test_dataset = load_data()
    model_name_path = model_name.replace('/', '_')  # 路径中不能包含'/'，所以替换为'_'
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./fine_tune_for_Seq2Seq/results/'+model_name_path+'_results',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=1,  # 修改批量大小以适应CPU
        per_device_eval_batch_size=1,  # 修改评估批量大小
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./fine_tune_for_Seq2Seq/logs',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # CPU不支持fp16
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = CustomTrainer(  # 使用自定义的Trainer
        model=model,  
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        data_collator=data_collator,
        tokenizer=tokenizer,  # tokenizer used for encoding the data
    )
    trainer.train()

trainer=train_model()

# 评估模型
trainer.evaluate()

# 预测
prompt = "Please appreciate this metaphor: 'Life is like a cup of coffee, with sweetness in bitterness'"
generated_answer = inference(prompt)
print(generated_answer)
