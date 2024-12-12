from utils import get_parser, get_dataloader, get_all_answer, label_loader,get_truth_contrast_dataloader,get_all_hidden_truth
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests
from PIL import Image
import torch
import numpy as np
import json 
import os
def main(args):
    # Set up the model and data
    print("Loading model")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    model = model.to(args.device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor.tokenizer.padding_side = "left"
    print("Loading dataloader")
    dataloader = get_truth_contrast_dataloader(args.dataset_name, args.split, processor, num_examples=args.num_examples, device=args.device)

    
    # Get the hidden imbedding
    print("Generating answers")
    #all_hidden_h, all_hidden_t,all_ans_h,all_ans_t,all_id_h,all_id_t,all_category_h,all_category_t = get_all_hidden(model, dataloader,processor,args)
    all_hidden_pos, all_hidden_neg,all_ans,all_id,all_category = get_all_hidden_truth(model, dataloader,processor,args)
    #save results 
    results={"pos_hidden":all_hidden_pos,"neg_hidden":all_hidden_neg,"all_ans":all_ans,
             "all_id":all_id,"all_category":all_category}
    
    if not os.path.exists(args.save_hidden_dir):
        os.makedirs(args.save_hidden_dir)
    torch.save(results, os.path.join(args.save_truth_hidden_dir,'POPE_truth_hidden.pt'))
     
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
