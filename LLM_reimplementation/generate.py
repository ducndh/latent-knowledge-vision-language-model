from utils import get_parser, load_model, get_dataloader, get_all_hidden_states, save_generations,get_all_output

def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, model_type = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    dataloader = get_dataloader(args.dataset_name, args.split, args.prompt_name, tokenizer, args.prompt_idx, batch_size=args.batch_size, 
                                num_examples=args.num_examples, model_type=model_type, use_decoder=args.use_decoder, device=args.device,text_key=args.text_key,label_key=args.label_key)

    # Get the hidden states and labels
    print("Generating hidden states")
    neg_hs, pos_hs, y = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers, 
                                              token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder)   
    #exm = get_all_output(model,tokenizer, dataloader, layer=args.layer, all_layers=args.all_layers, 
    #                                          token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder)
    # Save the hidden states and labels
    print("Saving hidden states")
    save_generations(neg_hs, args, generation_type="negative_hidden_states")
    save_generations(pos_hs, args, generation_type="positive_hidden_states")
    save_generations(y, args, generation_type="labels")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model_list=['Qwen/Qwen2.5-3B-Instruct','meta-llama/Llama-3.2-3B-Instruct']
    dataset_list=['imdb','amazon_polarity','ag_news','dbpedia_14','multi_nli',["glue","qnli"],"gimmaru/story_cloze-2016","piqa"]
    prompt_list=['imdb','amazon_polarity','ag_news','dbpedia_14','multi_nli',["glue","qnli"],"story_cloze/2016","piqa"]
    for i in model_list:
        #args.model_name=i
        main(args)
