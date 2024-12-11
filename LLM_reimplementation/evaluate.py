from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS,save_generations
import pandas as pd
import os
import numpy as np
def main(args, generation_args):
    # load hidden states and labels
    #model_list=['Qwen/Qwen2.5-3B-Instruct','meta-llama/Llama-3.2-3B-Instruct']
    #dataset_list=['imdb','amazon_polarity','ag_news','dbpedia_14','ag_news','multi_nli',"gimmaru/story_cloze-2016","piqa"]
    #split_list=['test','test','test','test','test','validation_mismatched','test','train']
    
    model_list = args.models
    dataset_list = args.datasets
    split_list = args.splits
    
    names=[]
    names.append('layer')
    for i in range(5):
        names.append('Logi{}'.format(i))
    for i in range(5):
        names.append('CCS{}'.format(i))
        
    #run experiment on every dataset
    for k in range(len(dataset_list)):
        generation_args.dataset_name=dataset_list[k]
        generation_args.split=split_list[k]
        
        #run experiment on both model
        for j in model_list:
            generation_args.model_name=j
            neg_hs_all, pos_hs_all, y = load_all_generations(generation_args)
            print(generation_args.dataset_name)
            min_y=min(y)
            max_y=max(y)
            y[np.where(y==min_y)]=0
            y[np.where(y==max_y)]=1
            
            # Make sure the shape is correct
            assert neg_hs_all.shape == pos_hs_all.shape
            logi_performance=np.zeros((neg_hs_all.shape[-1],5))
            CCS_performance=np.zeros((neg_hs_all.shape[-1],5))
            layer=np.zeros((neg_hs_all.shape[-1],1))
            
            #run experiment on every layer
            for i in range(neg_hs_all.shape[-1]):
            #for i in range(2):
                neg_hs, pos_hs = neg_hs_all[..., i], pos_hs_all[..., i]  # take the last layer
                if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
                    neg_hs = neg_hs.squeeze(1)
                    pos_hs = pos_hs.squeeze(1)

                
                layer[i]=i
                #for every model, dataset, and layer, we run CCS and logistic regression 5 times
                for g in range(5):
                    # Very simple train/test split (using the fact that the data is already shuffled)
                    # shuffle the train/test data every training cycle
                    id=np.arange(len(neg_hs))
                    np.random.shuffle(id)
                    neg_hs_train, neg_hs_test = neg_hs[id[:len(neg_hs) // 2]], neg_hs[id[len(neg_hs) // 2:]]
                    pos_hs_train, pos_hs_test = pos_hs[id[:len(pos_hs) // 2]], pos_hs[id[len(pos_hs) // 2:]]
                    y_train, y_test = y[id[:len(y) // 2]], y[id[len(y) // 2:]]

                    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
                    # you can also concatenate, but this works fine and is more comparable to CCS inputs
                    x_train = neg_hs_train - pos_hs_train  
                    x_test = neg_hs_test - pos_hs_test
                    lr = LogisticRegression(class_weight="balanced")
                    lr.fit(x_train, y_train)
                    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))
                    
                    logi_performance[i,g]=lr.score(x_test, y_test)
                    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
                    ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                                    var_normalize=args.var_normalize)
                    
                    # train and evaluate CCS
                    ccs.repeated_train()
                    ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
                    print("CCS accuracy: {}".format(ccs_acc))
                    CCS_performance[i,g]=ccs_acc
                    
            whole_data=np.concatenate((layer,logi_performance),axis=1)
            whole_data=np.concatenate((whole_data,CCS_performance),axis=1)
            data=pd.DataFrame(whole_data,columns=names)

            filename = 'results_{}_{}.csv'.format(generation_args.model_name,generation_args.dataset_name)
            filename = filename.replace('/','_')
            dir=os.path.join(os.getcwd(),'performance_instruct')
            if not os.path.exists(dir):
                os.makedirs(dir)
            data.to_csv(os.path.join(dir,filename))
        


if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")

    args = parser.parse_args()
    main(args, generation_args)
