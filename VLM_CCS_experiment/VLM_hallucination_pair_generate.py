from utils import get_parser, get_dataloader, get_all_answer, label_loader,get_contrast_dataloader,get_all_hidden
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests
from PIL import Image
import torch
import numpy as np
import json 

def main(args):
    # Set up the model and data
    print("Loading model")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    model = model.to(args.device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor.tokenizer.padding_side = "left"
    print("Loading dataloader")
    id_label=label_loader(args.label_dir)
    dataloader = get_contrast_dataloader(id_label,args.dataset_name, args.split, processor, num_examples=args.num_examples, device=args.device)

    
    # Get the hidden imbedding
    print("Generating answers")
    all_hidden_h, all_hidden_t,all_ans_h,all_ans_t,all_id_h,all_id_t,all_category_h,all_category_t = get_all_hidden(model, dataloader,processor,args)
    
    #save results 
    results={"hallucination_hidden":all_hidden_h,"normal_hidden":all_hidden_t,"all_ans_h":all_ans_h,"all_ans_t":all_ans_t,
             "all_id_h":all_id_h,"all_id_t":all_id_t,"all_category_h":all_category_h,"all_category_t":all_category_t}
    
    torch.save(results, 'experiment data/POPE_hidden_results_full.pt')
     
     
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
