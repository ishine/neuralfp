import os
import random
import numpy as np
import torch
import argparse
import faiss
import json
import shutil
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
parser.add_argument('--clean', default=False, type=bool,
                    help='organize test data into a single directory')


# Directories
root = os.path.dirname(__file__)

data_dir = os.path.join(root,"data")
ir_dir = os.path.join(root,'data/ir_filters')
noise_dir = os.path.join(root,'data/noise')
fp_dir = os.path.join(root,'fingerprints')
result_dir = os.path.join(root,'results')


device = torch.device("cuda")

def load_index(dirpath):
    dataset = {}
    idx = 0
    json_path = os.path.join(data_dir, dirpath.split('/')[-1] + ".json")
    
    if not os.path.exists(json_path):
        for filename in os.listdir(dirpath):
          if filename.endswith(".wav") or filename.endswith(".mp3"): 
            dataset[idx] = filename
            idx += 1
        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    return json_path


def clean_data(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    for sub in subfolders:
        for f in os.listdir(sub):
            src = os.path.join(sub, f)
            dst = os.path.join(folder, f)
            shutil.move(src, dst)
        shutil.rmtree(sub)
    
    test = os.listdir(folder)
    for item in test:
        if not (item.endswith(".wav") or item.endswith(".mp3")):
            os.remove(os.path.join(folder, item))
            
def create_fp_db(dataloader, model):
    fp_db = {}
    print("=> Creating fingerprints...")
    for idx, (x,fname) in enumerate(dataloader):
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

def compute_sequence_search(D,I):
    D = 1 - np.array(D)
    I = np.array(I)
    I_pr = []
    for i,r in enumerate(D):
        I_pr.append([(x - i) for x in I[i]])
    I_pr = np.array(I_pr)
    cdt = np.unique(I_pr.flatten())
    score = []
    I_flat = I.flatten()
    D_flat = D.flatten()
    
    for idx in cdt:
        pos = np.where((I_flat >= idx) & (I_flat <= idx + len(D)))[0]
        score.append(np.sum(D_flat[pos]))  

    return cdt[np.argmax(score)], str(round(np.max(score), 3))

def evaluate_hitrate(ref_db, query_db):
    audio_idx = 0
    ref_list = []
    
    for i in range(len(ref_db)):
        audio_idx += list(ref_db.values())[i].size(0)
        ref_list.append(audio_idx)
        
    print("=> Sanity check performed on random query")
    
    d = 128
    r = random.randrange(len(query_db))

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
    # min_id = np.argmin(D.flatten())
    # ix = I.flatten()[min_id]
    ix , sc = compute_sequence_search(D, I)
    
    # print("id: ",ix)
    # print(np.where(np.array(ref_list)>ix))
    
    idx = np.where(np.array(ref_list)>ix)[0][0]
    
    print("name of file from database",list(ref_db.keys())[idx])
    print("name of query file:\n",list(query_db.keys())[r])
    
    print("Computing hit-rate...")
    
    hit_song = 0
    hit_seg = 0
    result = {}
    for i in range(len(query_db)):
        xq = list(query_db.values())[i].cpu().numpy()
        D, I = index.search(xq, k)
        # min_id = np.argmin(D.flatten())
        # id = I.flatten()[min_id]
        id , sc = compute_sequence_search(D, I)
        idx = np.where(np.array(ref_list)>id)[0][0]
        
      
        query_name = list(query_db.keys())[i].split('.wav')[0].split('-')[0]
        # offset = int(list(query_db.keys())[i].split('.wav')[0].split('-')[-1])
        if list(ref_db.keys())[0].endswith(".mp3"):
            db_name = list(ref_db.keys())[idx].split('.mp3')[0]
        else:
            db_name = list(ref_db.keys())[idx].split('.wav')[0]
        if query_name == db_name:
              hit_song += 1
        # result[str(idx)] = sc      
    print(f"Hit rate = {hit_song}/{len(query_db)}")
    print(hit_song*1.0/len(query_db))
    
    return result




def main():
    args = parser.parse_args()
    if not os.path.exists(fp_dir):
        os.mkdir(fp_dir)
        
    if args.clean:
        clean_data(args.test_dir)
        
    if args.fp_path == '':
        checkpoint_dir = os.path.join(root, args.model_path)
        checkpoint = torch.load(checkpoint_dir)
        model = Neuralfp(encoder=encoder.Encoder()).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loading pre-trained model")
        # json_dir = load_index(args.test_dir)
        json_dir = 'data/1K_subset.json'
        test_dataset = NeuralfpDataset(path=args.test_dir, json_dir=json_dir, validate=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)
        # print(next(iter(test_loader))[0].shape)
        ref_db = create_fp_db(test_loader, model)
        torch.save(ref_db, os.path.join(fp_dir, args.test_dir.split('/')[-1] + "_ver7.pt"))

        
        
    elif args.fp_path != '' and  os.path.isfile(args.fp_path):
        ref_db = torch.load(args.fp_path)
        
    else:
        print("=>no fingeprint database'{}'".format(args.fp_path))

    if args.eval and os.path.isdir(args.query_dir):
        checkpoint_dir = os.path.join(root, args.model_path)
        checkpoint = torch.load(checkpoint_dir)
        model = Neuralfp(encoder=encoder.Encoder()).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        json_dir = load_index(args.query_dir)
        query_dataset = NeuralfpDataset(path=args.query_dir, json_dir=json_dir, validate=True)
        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)
        query_db = create_fp_db(query_loader, model)
        torch.save(query_db, os.path.join(fp_dir, args.query_dir.split('/')[-1] + "_ver7.pt"))
        
        result = evaluate_hitrate(ref_db, query_db)
        result_path = os.path.join(result_dir, args.query_dir.split('/')[-1] + "_ver7.json")
        with open(result_path, 'w') as fp:
            json.dump(result, fp)


if __name__ == '__main__':
    main()
            