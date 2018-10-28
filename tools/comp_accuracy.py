import matplotlib.pyplot as plt
import pickle, sys, numpy as np, json, argparse, h5py, glob, os
from prettytable import PrettyTable
import sys
sys.path.append('data_tools/')
import utils_prep

def get_rels(sg_entries, opposite_lab):
    gt_rels = np.zeros((0,3+1)) # sg_entry_id + o1_idx + o2_idx +  number of predicate category
    scores_rels = np.zeros((0,1))
    scores_raw = np.zeros((0,1))
    for (i,(sg_entry, roidb)) in enumerate(sg_entries):
        gt_rel = roidb['gt_relations']
        gt_rels = np.vstack((gt_rels, np.hstack(( np.ones((gt_rel.shape[0],1))*i ,gt_rel)) ))
        pred_rels = sg_entry['relations'] #        pred_ranked = np.argsort(-sg_entry['relations'], axis=2)
        for rel in gt_rel:
            if pred_rels[rel[0],rel[1],rel[2]] > pred_rels[rel[0],rel[1],opposite_lab[rel[2]]]:
                scores_rels = np.vstack((scores_rels, (rel[2]) ))
            else:
                scores_rels = np.vstack((scores_rels, (opposite_lab[rel[2]]) ))
    return (scores_rels, gt_rels)
   
def get_opposite_labels(label_dict):
    opposite_labels = {}
    for i in label_dict.keys():
        for j in label_dict.keys():
            if j == 'not_' + i:
                opposite_labels[ int(label_dict[i])] = int(label_dict[j])
                opposite_labels[int(label_dict[j])] = int(label_dict[i])
    return opposite_labels

def case_for_rel(gt_rels,cases, outliers):
    nrel = gt_rels.shape[0]
    case_rels = np.ones((nrel,1))*(-2)
    outlier_rels = np.ones((nrel,1))*(-2) 
    for r in range(nrel):
        entry = int(gt_rels[r,0])
        case_rels[r] = cases[entry]
        outlier_rels[r] = outliers[entry]
    return case_rels, outlier_rels

def case_for_obj(gt_objs, cases, outliers):
    no = gt_objs.shape[0]
    case_objs = np.ones((no,1))*(-2)
    outlier_objs = np.ones((no,1))*(-2) 
    for r in range(no):
        entry = int(gt_objs[r,0])
        case_objs[r] = cases[entry]
        outlier_objs[r] = outliers[entry]
    return case_objs, outlier_objs

def op_l(l):
    if 'not_' in l:
        l = l.replace('not_','')
    else:
        l = 'not_' + l
    return l

def get_ac(seq2span, sg_entries, label_dict, sub):
    spans = sorted(seq2span.values())
    entry2span = dict([ (i, seq2span[roidb['seq_name']]) for (i,(sg_entry,roidb)) in enumerate(sg_entries) ])
    idx = range(1,len(spans),len(spans)/5)
    vals = np.zeros((len(idx)-1,))
    for i in range(len(idx)-1):
        s1 = spans[idx[i]]
        s2 = spans[idx[i+1]]
        idx_entries = [ e for (e,s) in entry2span.items() if s1 < s and s < s2 ]
        idx_rel = [ r for (r,(e,o1,o2,p)) in enumerate(gt_rels) if e in idx_entries and (label_dict['predicate_to_idx'][sub]==p or label_dict['predicate_to_idx'][op_l(sub)]==p) ]
        vals[i] = np.mean(gt_rels[idx_rel,3]== pred_rels[idx_rel,0])
    return vals
   
def acc_per_pred(ygt, ypred, label_dict):
    predicates = [ p for p in label_dict['predicate_to_idx'].keys() if 'not_' not in  p ]
    pred2ac = {}
    for p in predicates:
        idx = np.logical_or(ygt==label_dict['predicate_to_idx'][p], ygt == label_dict['predicate_to_idx'][op_l(p)])
        pred2ac[p] = np.mean(ygt[idx]==ypred[idx])
    return pred2ac

def get_objs(sg_entries):
    gts = np.zeros((0,3))
    preds = np.zeros((0,2))
    for (i,(sg_entry, roidb)) in enumerate(sg_entries):
        sys.path.append('lib')
        from datasets.eval_utils import ground_predictions
        roidb['boxes'] = roidb['boxes'] * 0.6198347107486631
        #gt_to_pred = roidb['gt_to_pred_object']
        gt_to_pred = ground_predictions(sg_entry, roidb, 0.5)
        for (i,v) in gt_to_pred.items():
            if i!=v:
                import pdb; pdb.set_trace() #assert(i==v)
        preds_o = np.argmax(sg_entry['scores'],axis=1)
        for (op, ogt) in gt_to_pred.items():
            gt = roidb['gt_classes'][ogt]
            gts = np.vstack((gts, np.array((i,ogt,gt))))
            pred = preds_o[op]
            preds = np.vstack((preds,np.array((op, pred))))
    return (preds, gts)

   
def acc_from_file(f):
    sg_entries = pickle.load(open(f))
    global opposite_labels
    (pred_rels, gt_rels) = get_rels(sg_entries, opposite_labels)
    ygt = gt_rels[:,3]
    ypred = pred_rels[:,0]
    pred2ac = acc_per_pred(ygt, ypred, label_dict)
    (pred_objs, gt_objs) = get_objs(sg_entries)
    m_objs = np.mean(pred_objs[:,1]==gt_objs[:,2])
    pred2ac['object'] = m_objs
    return pred2ac


def get_acc(result_dir):
    rs = {}
    iters = []
    rs['split_0'] = {}
    rs['split_2'] = {}
    for train_dump_file in glob.glob(os.path.join(result_dir,'*_0.pc')):
        print train_dump_file
        it = int(train_dump_file.split('_')[-2]) 
        train_acc = acc_from_file(train_dump_file)
        rs['split_0'][it] = train_acc
        test_dump_file = train_dump_file.replace('_0.','_2.')
        test_acc = acc_from_file(test_dump_file)
        rs['split_2'][it] = test_acc
    return rs


parser = argparse.ArgumentParser(description='evaluate the accuracy of a passed experiment in multi label setting')
parser.add_argument('--result_dir')
parser.add_argument('--dict_file')
parser.add_argument('--output')
args = parser.parse_args()

# loading the data
result_dir = args.result_dir
dict_file = args.dict_file
label_dict =  json.load(open(dict_file)) 
opposite_labels = get_opposite_labels(label_dict['predicate_to_idx'])
rs = get_acc(result_dir)

pickle.dump(rs, open(args.output,'wb'))
