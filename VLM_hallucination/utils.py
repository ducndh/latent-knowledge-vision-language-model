import os
import functools
import argparse
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset
from transformers import pipeline
#from torchview import draw_graph

def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--dataset_name", nargs='+', default="imdb", help="Name of the dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Which split of the dataset to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=9000, help="Number of examples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=1, help="Number of max generated tokens")
    parser.add_argument("--save_logits_dir", type=str, default=None, help="dir of logits and label file")
    parser.add_argument("--save_hidden_dir", type=str, default=None, help="dir of hidden embedding file")
    parser.add_argument("--save_truth_hidden_dir", type=str, default=None, help="dir of truth contrast pair hidden embedding file")
    
    #parser.add_argument("--file_dir", type=str, default='experiment data/POPE_truth_hidden_results_full.pt')
    #parser.add_argument("--repeat_times", type=int, default=1, help="Number of repeating times for every sample")
    
    return parser


class ChatDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with a collection of prompts for that dataset from promptsource and a corresponding prompt index, 
    returns a dataset that creates contrast pairs using that prompt
    
    Truncates examples larger than max_len, which can mess up contrast pairs, so make sure to only give it examples that won't be truncated.
    """
    def __init__(self, raw_dataset, processor, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.device = device
        

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]
        question, answer,image, id ,category= data["question"], data["answer"], data["image"], data["id"],data["category"]
        #image = Image.open(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        #inputs = self.processor(images=image, text=prompt, padding=True, return_tensors="pt")
        
        return image, prompt, answer, id,category


class ContrastDataset(Dataset):
    """
    Given a full dataset and the indexs of hallucination samples, return a contrast pair dataset. Every pair contains a hallucination prompt and normal prompt and a same image. 
    """
    def __init__(self, raw_dataset, hallu_idxs, id_label, processor, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.hallu_dataset = hallu_idxs
        self.id_label=id_label
        self.processor = processor
        self.device = device
        

    def __len__(self):
        return len(self.hallu_dataset)

    def __getitem__(self, index):
        # get the original example
        data_hallu = self.raw_dataset[int(index)]
        question, answer,image, image_source, id ,category= data_hallu["question"], data_hallu["answer"], data_hallu["image"], data_hallu["image_source"], data_hallu["id"],data_hallu["category"]
        id=int(id)
        if id%2==0:
            true_index=int(id)+1
        else:
            true_index=int(id)-1
        data_true = self.raw_dataset[true_index]
        question_t, answer_t,image_t, image_source_t, id_t ,category_t= data_true["question"], data_true["answer"], data_true["image"], data_true["image_source"], data_true["id"],data_true["category"]
        
        #search leftward and rightward to find a normal prompt with the same image as the hallucination prompt.
        true_index_cp=true_index
        while self.id_label[true_index_cp,0]!=1:
            if true_index_cp-1>=0 and self.raw_dataset[true_index_cp-1]["image_source"]==image_source:
                true_index_cp-=1
            else:
                break
        if self.id_label[true_index_cp,0]!=1:
            true_index_cp=true_index
            while self.id_label[true_index_cp,0]!=1:
                if true_index_cp+1<9000 and self.raw_dataset[true_index_cp+1]["image_source"]==image_source:
                    true_index_cp+=1
                else:
                    break
        
        assert self.id_label[true_index_cp,0]==1, print("Cannot find the no hallucination pair for this data!")

        data_true = self.raw_dataset[true_index_cp]
        question_t, answer_t,image_t, image_source_t, id_t ,category_t= data_true["question"], data_true["answer"], data_true["image"], data_true["image_source"], data_true["id"],data_true["category"]
        
        #image = Image.open(image)
        conversation_hallu = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        conversation_true = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_t},
                ],
            },
        ]
        
        prompt_hallu = self.processor.apply_chat_template(conversation_hallu, add_generation_prompt=True)
        prompt_true = self.processor.apply_chat_template(conversation_true, add_generation_prompt=True)
        
        #inputs = self.processor(images=image, text=prompt, padding=True, return_tensors="pt")
        
        return image, prompt_hallu, answer, id,category, prompt_true, answer_t, id_t, category_t

class TruthContrastDataset(Dataset):
    """
    Given a full dataset and the indexs of hallucination samples, return a contrast pair dataset. Every pair contains a hallucination prompt and normal prompt and a same image. 
    """
    def __init__(self, raw_dataset, processor, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        #self.hallu_dataset = hallu_idxs
        #self.id_label=id_label
        self.processor = processor
        self.device = device
        

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]
        question, answer,image, image_source, id ,category= data["question"], data["answer"], data["image"], data["image_source"], data["id"],data["category"]
        
        
        if "Is there" in question:
            cap="Is there"
            rep="There is"
        elif "Are there" in question:
            cap="Are there"
            rep="There are"
        else:
            raise Exception("Cannot find is or are in the question!")
        pos_question=question
        neg_question=question
        #pos_question=pos_question.replace(cap,"Is the following description correct? {}".format(rep))
        #neg_question=neg_question.replace(cap,"Is the following description correct? {}".format(rep+' not'))
        pos_question=pos_question.replace(cap,rep)
        neg_question=neg_question.replace(cap,rep+' not')
        
        pos_question=pos_question[::-1].replace('?', '.', 1)[::-1]
        neg_question=neg_question[::-1].replace('?', '.', 1)[::-1]
        #image = Image.open(image)
        conversation_pos = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": pos_question},
                ],
            },
        ]

        conversation_neg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": neg_question},
                ],
            },
        ]
        
        prompt_pos = self.processor.apply_chat_template(conversation_pos, add_generation_prompt=True)
        prompt_neg = self.processor.apply_chat_template(conversation_neg, add_generation_prompt=True)
        
        #inputs = self.processor(images=image, text=prompt, padding=True, return_tensors="pt")
        
        return image, prompt_pos, prompt_neg, answer, id,category

def get_dataloader(dataset_name, split, processor,  num_examples=1000, device="cuda"):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    #raw_dataset = load_dataset(*dataset_name,trust_remote_code=True)[split]
    raw_dataset = load_dataset(*dataset_name,trust_remote_code=True)[split]

    # create the ConstrastDataset
    contrast_dataset = ChatDataset(raw_dataset, processor, device=device)
#    for i,[a,b,c] in enumerate(contrast_dataset):
#        print(i,a,b,c)
    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    random_idxs = np.random.permutation(len(contrast_dataset))
    random_idxs = random_idxs[:num_examples]
    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, random_idxs)
    #dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return subset_dataset

def get_contrast_dataloader(id_label,dataset_name, split, processor,  num_examples=1000, device="cuda"):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    raw_dataset = load_dataset(*dataset_name,trust_remote_code=True)[split]
    # get the hallucination dataset
    hallu_idxs=np.where(id_label==0)
    hallu_idxs=hallu_idxs[0]
    num_examples=hallu_idxs.shape[0]
    #hallu_set= torch.utils.data.Subset(raw_dataset, hallu_idxs)
    #hallu_set=raw_dataset[hallu_idxs]
    # create the ConstrastDataset
    contrast_dataset = ContrastDataset(raw_dataset,hallu_idxs, id_label, processor, device=device)
  
    #for i,a in enumerate(contrast_dataset):
    #    print(i,a)
    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    random_idxs = np.random.permutation(hallu_idxs)
    random_idxs = random_idxs[:num_examples]
    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, random_idxs)
    #dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return subset_dataset

def get_truth_contrast_dataloader(dataset_name, split, processor,  num_examples=1000, device="cuda"):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    raw_dataset = load_dataset(*dataset_name,trust_remote_code=True)[split]

    contrast_dataset = TruthContrastDataset(raw_dataset, processor, device=device)
    
    #for i,a in enumerate(contrast_dataset):
    #    print(i,a)
    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    random_idxs = np.random.permutation(len(contrast_dataset))
    random_idxs = random_idxs[:num_examples]
    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, random_idxs)
    #dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return subset_dataset

def get_all_answer(model, dataset, processor,args):
    """
    Given a model, a tokenizer, and a dataloader, returns the hidden states (corresponding to a given position index) in all layers for all examples in the dataloader,
    along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples *with a candidate label already added* to each example.
    E.g. this function should be used for "Q: Is 2+2=5? A: True" or "Q: Is 2+2=5? A: False", but NOT for "Q: Is 2+2=5? A: ".
    """
    all_pred_ans = []
    all_logits = []
    all_ans = []
    all_id = []
    all_category = []
    
    model.eval()
    images = []
    prompts = []
    answers = []
    diff = []
    for i,[image,prompt,answer,id,category] in enumerate(tqdm(dataset)):
        images.append(image)
        prompts.append(prompt)
        answers.append(answer)
        all_id.append(id)
        all_category.append(category)
        
        if (i+1) % args.batch_size == 0:
            inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model.device)
            #print(inputs['attention_mask'][0,:100])
            

            generate_ids = model.generate(**inputs,  output_logits =True, return_dict_in_generate=True,max_new_tokens=args.max_new_tokens)
            #generate_ids = model(**inputs,output_hidden_states=True)
            
            outputs=processor.batch_decode(generate_ids['sequences'], skip_special_tokens=True)
            all_pred_ans.append(outputs)
            all_logits.append(generate_ids.logits[0].detach().cpu())
            
            all_ans.append(answers)
            images = []
            prompts = []
            answers = []

    all_pred_ans_fl=[]
    all_ans_fl = []
    for i in range(len(all_pred_ans)):
        for j in range(args.batch_size):
            all_pred_ans_fl.append(all_pred_ans[i][j])
            all_ans_fl.append(all_ans[i][j])
            
    all_logits_fl = np.concatenate(all_logits,axis=0)
    
    return all_pred_ans_fl, all_ans_fl,all_logits_fl,all_id,all_category



def get_all_hidden(model, dataset, processor,args):
    """
    Given a model, a tokenizer, and a dataloader, returns the hidden states (corresponding to a given position index) in all layers for all examples in the dataloader,
    along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples *with a candidate label already added* to each example.
    E.g. this function should be used for "Q: Is 2+2=5? A: True" or "Q: Is 2+2=5? A: False", but NOT for "Q: Is 2+2=5? A: ".
    """
    all_hidden_h = []
    all_hidden_t = []
    all_ans_h = []
    all_ans_t = []
    all_id_h = []
    all_id_t = []
    all_category_h = []
    all_category_t = []
    
    model.eval()
    images = []
    prompts_h = []
    answers_h = []
    prompts_t = []
    answers_t = []
    
    
    for i,[image,prompt_h,answer_h,id_h,category_h,prompt_t,answer_t,id_t,category_t] in enumerate(tqdm(dataset)):
        images.append(image)
        prompts_h.append(prompt_h)
        answers_h.append(answer_h)
        all_id_h.append(id_h)
        all_category_h.append(category_h)
        all_ans_h.append(answer_h)
        
        prompts_t.append(prompt_t)
        answers_t.append(answer_t)
        all_id_t.append(id_t)
        all_category_t.append(category_t)
        all_ans_t.append(answer_t)
        
        if (i+1) % args.batch_size == 0:
            with torch.no_grad():
                inputs_h = processor(images=images, text=prompts_h, padding=True, return_tensors="pt").to(model.device)
                outputs_h= model(**inputs_h,  output_hidden_states=True)
            hs_tuple_h = outputs_h["hidden_states"]
            hs_h = torch.stack([h.detach().cpu() for h in hs_tuple_h], axis=-1)
            final_hs_h = hs_h[torch.arange(hs_h.size(0)), -1]
            all_hidden_h.append(final_hs_h)
            
            with torch.no_grad():
                inputs_t = processor(images=images, text=prompts_t, padding=True, return_tensors="pt").to(model.device)
                outputs_t= model(**inputs_t,  output_hidden_states=True)
            hs_tuple_t = outputs_t["hidden_states"]
            hs_t = torch.stack([h.detach().cpu() for h in hs_tuple_t], axis=-1)
            final_hs_t = hs_t[torch.arange(hs_t.size(0)), -1]
            all_hidden_t.append(final_hs_t)
            
            images = []
            prompts_h = []
            answers_h = []
            prompts_t = []
            answers_t = []
    
            
    all_hidden_h = np.concatenate(all_hidden_h,axis=0)
    all_hidden_t = np.concatenate(all_hidden_t,axis=0)
    
    return all_hidden_h, all_hidden_t,all_ans_h,all_ans_t,all_id_h,all_id_t,all_category_h,all_category_t

def get_all_hidden_truth(model, dataset, processor,args):
    """
    Given a model, a tokenizer, and a dataloader, returns the hidden states (corresponding to a given position index) in all layers for all examples in the dataloader,
    along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples *with a candidate label already added* to each example.
    E.g. this function should be used for "Q: Is 2+2=5? A: True" or "Q: Is 2+2=5? A: False", but NOT for "Q: Is 2+2=5? A: ".
    """
    all_hidden_pos = []
    all_hidden_neg = []
    all_ans = []
    all_id = []
    all_category = []
    
    model.eval()
    images = []
    prompts_pos = []
    prompts_neg = []
    
    
    for i,[image, prompt_pos, prompt_neg, answer, id,category] in enumerate(tqdm(dataset)):
        images.append(image)
        prompts_pos.append(prompt_pos)
        prompts_neg.append(prompt_neg)
        
        all_id.append(id)
        all_category.append(category)
        all_ans.append(answer)
        
        
        if (i+1) % args.batch_size == 0:
            with torch.no_grad():
                inputs_pos = processor(images=images, text=prompts_pos, padding=True, return_tensors="pt").to(model.device)
                outputs_pos= model(**inputs_pos,  output_hidden_states=True)
            hs_tuple_pos = outputs_pos["hidden_states"]
            hs_pos = torch.stack([h.detach().cpu() for h in hs_tuple_pos], axis=-1)
            final_hs_pos = hs_pos[torch.arange(hs_pos.size(0)), -1]
            all_hidden_pos.append(final_hs_pos)
            
            with torch.no_grad():
                inputs_neg = processor(images=images, text=prompts_neg, padding=True, return_tensors="pt").to(model.device)
                outputs_neg= model(**inputs_neg,  output_hidden_states=True)
            hs_tuple_neg = outputs_neg["hidden_states"]
            hs_neg = torch.stack([h.detach().cpu() for h in hs_tuple_neg], axis=-1)
            final_hs_neg = hs_neg[torch.arange(hs_neg.size(0)), -1]
            all_hidden_neg.append(final_hs_neg)
            
            images = []
            prompts_pos = []
            prompts_neg = []
    
            
    all_hidden_pos = np.concatenate(all_hidden_pos,axis=0)
    all_hidden_neg = np.concatenate(all_hidden_neg,axis=0)
    
    return all_hidden_pos, all_hidden_neg,all_ans,all_id,all_category



def label_loader(label_data):
    data_dict = torch.load(label_data)
    id_label=np.zeros((9000,1))

    for i in range(len( data_dict['gt_answer'])):
        ans=data_dict['generated_ans_only'][i]
        id=data_dict['sample_id'][i]
        id=int(id)
        if ans=='nan':
            id_label[id]=2
        if ans!='nan':
            if data_dict['gt_answer'][i]==data_dict['generated_ans_only'][i]:
                id_label[id]=1
            else:
                id_label[id]=0
                
    return id_label


def load_single_generation(args, generation_type="hidden_states"):
    # use the same filename as in save_generations
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
    filename = filename.replace('/','_')
    if len(filename)>150:
        filename=filename[:150]+'all_layers_{}'.format(args.all_layers)
    data=np.load(os.path.join(args.save_dir, filename+'.npy'))
    
    return data

def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    
    data=torch.load(args.save_hidden_dir)
    neg_hs = data['neg_hidden']
    #neg_hs=neg_hs.squeeze(1)
    pos_hs = data['pos_hidden']
    #pos_hs=pos_hs.squeeze(1)
    y = data['all_ans']
    return neg_hs, pos_hs ,y


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    

    def save_probe(self,path):
        torch.save(self.best_probe.state_dict(), path)
        
    def load_probe(self, path):
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)
        self.best_probe.load_state_dict(torch.load(path, weights_only=True))
        self.best_probe.eval()
        
    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss


    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss