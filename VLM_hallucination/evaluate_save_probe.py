from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations_hallu, CCS
import pandas as pd
import os
import numpy as np
import torch

import copy

def main(args, generation_args):
 
    neg_hs_all, pos_hs_all = load_all_generations_hallu(generation_args)
    print(generation_args.dataset_name)
    
    # Make sure the shape is correct
    assert neg_hs_all.shape == pos_hs_all.shape
    logi_performance=np.zeros((neg_hs_all.shape[-1],5))
    CCS_performance=np.zeros((neg_hs_all.shape[-1],5))
    layer=np.zeros((neg_hs_all.shape[-1],1))
    
    #run experiment on every layer
    for i in range(neg_hs_all.shape[-1]-8,neg_hs_all.shape[-1]):

        neg_hs, pos_hs = neg_hs_all[..., i], pos_hs_all[..., i]  # take the ith layer
        if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
            neg_hs = neg_hs.squeeze(1)
            pos_hs = pos_hs.squeeze(1)

        
        layer[i]=i
        #for every model, dataset, and layer, we run CCS and logistic regression 5 times
        for g in range(1):
            # Very simple train/test split (using the fact that the data is already shuffled)
            # shuffle the train/test data every training cycle
            id=np.arange(len(neg_hs))
            np.random.shuffle(id)
            
            neg_hs_cp=copy.deepcopy(neg_hs)
            pos_hs_cp=copy.deepcopy(pos_hs)
            
            
            #because all samples in neg_hs are hallucination ones, so first exchange the first half with pos_hs and then shuffle.
            flip_data=copy.deepcopy(neg_hs_cp[:len(neg_hs_cp) // 2])
            neg_hs_cp[:len(neg_hs_cp) // 2]=copy.deepcopy(pos_hs_cp[:len(pos_hs_cp) // 2])
            pos_hs_cp[:len(pos_hs_cp) // 2]=flip_data
            y=np.hstack((np.ones(len(neg_hs_cp)//2,dtype=int),np.zeros(len(neg_hs_cp)-len(neg_hs_cp)//2,dtype=int)))
            
            neg_hs_train, neg_hs_test = neg_hs_cp[id[:len(neg_hs_cp) // 2]], neg_hs_cp[id[len(neg_hs_cp) // 2:]]
            pos_hs_train, pos_hs_test = pos_hs_cp[id[:len(pos_hs_cp) // 2]], pos_hs_cp[id[len(pos_hs_cp) // 2:]]
            y_train, y_test = y[id[:len(y) // 2]], y[id[len(y) // 2:]]
            
            x_train = neg_hs_train - pos_hs_train
            x_test = neg_hs_test - pos_hs_test
            

            # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
            # you can also concatenate, but this works fine and is more comparable to CCS inputs
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
            
            #save the ccs probe
            dir=args.save_probe_dir
            if not os.path.exists(dir):
                os.makedirs(dir)
            probe_path = os.path.join(dir,'probe_from_layer{}.pt'.format(i))
            ccs.save_probe(probe_path)
            
            ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
            print("CCS accuracy: {}".format(ccs_acc))

    

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
