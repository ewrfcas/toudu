import re
import numpy as np
i = 0
np.random.seed(102131)
new_data = []
neg_num = 0

def get_subsens(context):
    sentences = re.split('[?？。“”！!…,，]', context)
    sen_count = {}
    for s in sentences:
        if s not in sen_count:
            sen_count[s] = 1
        else:
            sen_count[s] += 1
    sentences_ = []
    for s in sentences:
        if '[BLANK' not in s and s != '' and len(s) >= 10 and len(s) < 30 and sen_count[s] <= 1:
            can_append = True
            # 保证不被其他包含
            for s_ in sen_count:
                if s != s_ and s in s_:
                    can_append = False
                    break
            if can_append:
                sentences_.append(s)
    sentences = sentences_
    
    return sentences
    
for context in contexts:
    # 分句
    context = context.strip()
    if context=="":
        continue
    sentences = re.split('(。|！|\!|？|\?|“|”)',context)
    sentences = [s for s in sentences if s!=""]
    new_contexts = []
    i_sen = 0
    
    if np.random.random() < 0.6 and len(context)>1000:
        max_token_lengths = np.random.randint(600, 1200)
        while i_sen<len(sentences):
            new_contexts.append("")
            while (len(new_contexts[-1])<max_token_lengths and i_sen<len(sentences)) or new_contexts[-1][-1] in {':','：'}:
                new_contexts[-1]+=(sentences[i_sen])
                i_sen+=1
            if i_sen<len(sentences) and sentences[i_sen] in {"。","!",'！','?','？','”','…'}:
                new_contexts[-1]+=sentences[i_sen]
                i_sen+=1
    else:
        new_contexts = [context]
        
    total_sentences = get_subsens(context)
    total_all_sentences = re.split('[?？。“”！!…,，]', context)
    total_all_sentences = [s for s in total_all_sentences if s!=""]
    
    for context_ in new_contexts:
        in_sentences = [s for s in total_sentences if s in context_]
        out_sentences = [s for s in total_sentences if s not in context_]
        random_num = min(np.random.randint(min(int(len(in_sentences) * 2/3),15), 16), len(in_sentences))
        # 开始一个个替换掉
        rd_sen = np.random.choice(in_sentences, size=random_num, replace=False)
        choices = []
        pre_mask = False
        for sen in total_all_sentences:
            if sen in rd_sen and (pre_mask is False or np.random.random()<0.5):
                choices.append(sen)
                pre_mask=True
            else:
                pre_mask=False
        answers = []
        for i in range(len(choices)):
            context_ = context_.replace(choices[i], '[BLANK' + str(i + 1) + ']')
            answers.append(i)
            
        # 加入负样本
        if len(out_sentences)>0:
            neg_num+=1
            random_out_num = min(len(out_sentences), np.random.randint(2, 4))
            rd_sen_out = np.random.choice(out_sentences, size=random_out_num, replace=False)
            for sen in rd_sen_out:
                choices.append(sen)
                answers.append(-1)

        new_data.append({'context_id':'aug_'+str(i),
                        'answers':answers,
                        'context':context_,
                        'choices':choices})    
