from utils import get_parser, get_dataloader, get_all_answer, label_loader
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
    #id_label=label_loader(args.label_data)
    dataloader = get_dataloader(args.dataset_name, args.split, processor, num_examples=args.num_examples, device=args.device)

    # Get the answers
    print("Generating answers")
    pred_ans,ans,logits,ids,categoties = get_all_answer(model, dataloader,processor,args)
    
    #print(pred_ans)
    
    ans_only=[]
    tot=0
    n_nan=0
    diff=0
    
    for i,w in enumerate(pred_ans):
        
        ans_word=w.split("ASSISTANT: ")
        last_word=ans_word[-1].lower()
            #ans_word=w.split("ASSISTANT: ")
        if len(ans_word)==1:
            last_word="nan"
            n_nan+=1
            
        if last_word==ans[i]:
            tot+=1
        ans_only.append(last_word)
        
                    
    print(tot/(len(ans)-n_nan))
    print(n_nan)
    #save results 
    results={"generated_answer":pred_ans,"gt_answer":ans,"generated_ans_only":ans_only,"logits":logits,"sample_id":ids,"category":categoties}
    
    torch.save(results, 'experiment data/POPE_results_full.pt')
     
     
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
