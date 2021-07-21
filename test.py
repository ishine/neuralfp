import os
import random
import numpy as np
import torch
import argparse
import faiss
import json

# Neuralfp
from neuralfp.modules.data import NeuralfpDataset
from neuralfp.modules.transformations import TransformNeuralfp
from neuralfp.neuralfp import Neuralfp
import neuralfp.modules.encoder as encoder


parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--test_dir', default='', type=str, metavar='PATH',
                    help='directory containing test dataset')
parser.add_argument('--fp_path', default='', type=str, metavar='PATH',
                    help='pre-computed fingerprint dataset path')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                    help='path for pre-trained model')
parser.add_argument('--query_dir', default='', type=str, metavar='PATH',
                    help='directory containing query dataset')
parser.add_argument('--eval', default=False, type=bool,
                    help='flag for evaluating query search')


# Directories
root = os.path.dirname(__file__)

data_dir = os.path.join(root,"data")
ir_dir = os.path.join(root,'data/ir_filters')
noise_dir = os.path.join(root,'data/noise')
fp_dir = os.path.join(root,'fingerprints')


device = torch.device("cuda")

def load_index(dirpath):
    dataset = {}
    idx = 0
    json_path = os.path.join(data_dir, dirpath.split('/')[-1] + ".json")
    if not os.path.isfile(json_path):
        for filename in os.listdir(dirpath):
          if filename.endswith(".wav") or filename.endswith(".mp3"): 
            dataset[idx] = filename
            idx += 1
        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    return json_path
        
def create_fp_db(dataloader, model):
    fp_db = {}
    print("=> Creating fingerprints...")
    for idx, (x,fname) in enumerate(dataloader):
        print(fname)
        fp = []
        splits = x[0]
        for s in splits:
            inp = torch.unsqueeze(s,0).to(device)
            with torch.no_grad():
                _,_,z,_= model(inp,inp)
            z = z.detach()
            fp.append(z)
        qname = fname[0]
        if idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z.shape}\t file: {qname}")
        fp = torch.cat(fp)
        fp_db[qname] = fp
    
    return fp_db

def evaluate_hitrate(ref_db, query_db):
    audio_idx = 0
    ref_list = []
    
    for i in range(len(ref_db)):
        audio_idx += list(ref_db.values())[i].size(0)
        ref_list.append(audio_idx)
        
        print("=> Sanity check performed on random query")
        
        d = 128
        r = random.randrange(len(os.listdir(query_db)))

        xb = torch.cat(list(ref_db.values())).cpu().numpy()
        xq = list(query_db.values())[r].cpu().numpy()
          
        nlist = 100
        m = 8
        k = 4
        quantizer = faiss.IndexFlatL2(d) 
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        
        index.train(xb)
        index.add(xb)
        
        index.nprobe = 20 
        
        D, I = index.search(xq, k)
        print("k-NN distance and index vectors:")
        print(D)
    
        print(I)
        min_id = np.argmin(D.flatten())
        id = I.flatten()[min_id]
        
        idx = np.where(np.array(ref_list)>id)[0][0]
        
        print("name of file from database",list(ref_db.keys())[idx])
        print("name of query file:\n",list(query_db.keys())[r])
        
        print("Computing hit-rate...")
        
        hit = 0
        for i in range(len(query_db)):
            xq = list(query_db.values())[r].cpu().numpy()
            D, I = index.search(xq, k)
            min_id = np.argmin(D.flatten())
            id = I.flatten()[min_id]
            idx = np.where(np.array(ref_list)>id)[0][0]
          
            query_name = list(query_db.keys())[r].split('.wav')[0].split('-')[0]
            db_name = list(ref_db.keys())[idx].split('.wav')[0]
            if query_name == db_name:
                  hit+=1
                  
        print("Hit rate = {hit}/{len(query_db)}")


def main():
    args = parser.parse_args()
    if not os.path.exists(fp_dir):
        os.mkdir(fp_dir)
        
    if args.fp_path == '':
        checkpoint_dir = os.path.join(root, args.model_path)
        checkpoint = torch.load(checkpoint_dir)
        model = Neuralfp(encoder=encoder.Encoder()).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loading pre-trained model")
        json_dir = load_index(args.test_dir)
        test_dataset = NeuralfpDataset(path=args.test_dir, json_dir=json_dir, validate=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)
        ref_db = create_fp_db(test_loader, model)
        print(list(ref_db.keys())[0:10])
        torch.save(ref_db, os.path.join(fp_dir, args.test_dir.split('/')[-1] + "_aug1.pt"))

        
        
    elif args.fp_path != '' and  os.path.isfile(args.fp_path):
        ref_db = torch.load(args.fp_path)
        
    else:
        print("=>no fingeprint database'{}'".format(args.fp_path))

    if args.eval and os.path.isdir(args.query_dir):
        model = Neuralfp(encoder=encoder.Encoder()).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        json_dir = load_index(args.query_dir)
        query_dataset = NeuralfpDataset(path=args.test_dir, json_dir=json_dir, validate=True)
        query_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)
        query_db = create_fp_db(query_loader, model)
        torch.save(query_db, os.path.join(fp_dir, args.query_dir.split('/')[-1] + "aug1.pt"))
        
        evaluate_hitrate(ref_db, query_db)
        

if __name__ == '__main__':
    main()
            