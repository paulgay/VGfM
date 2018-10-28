import pickle, sys, numpy as np, json, argparse, h5py
from prettytable import PrettyTable

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
   
def get_objs(sg_entries):
    gts = np.zeros((0,3))
    preds = np.zeros((0,2))
    for (i,(sg_entry, roidb)) in enumerate(sg_entries):
        gt_to_pred = roidb['gt_to_pred_object']
        preds_o = np.argmax(sg_entry['scores'],axis=1)
        for (op, ogt) in gt_to_pred.items():
            gt = roidb['gt_classes'][ogt]
            gts = np.vstack((gts, np.array((i,ogt,gt))))
            pred = preds_o[op]
            preds = np.vstack((preds,np.array((op, pred))))
    return (preds, gts)

def get_opposite_labels(label_dict):
    opposite_labels = {}
    for i in label_dict.keys():
        for j in label_dict.keys():
            if j == 'not_' + i:
                opposite_labels[ int(label_dict[i])] = int(label_dict[j])
                opposite_labels[int(label_dict[j])] = int(label_dict[i])
    return opposite_labels

def get_cases(imdb, sg_entries, prop):
    im_paths = imdb['im_paths'][()]
    rpn_to_imdb = prop['im_to_imdb_idx'][()]
    nentries = len(sg_entries)
    cases = np.ones((nentries,1))*(-2)
    outliers = np.ones((nentries,1))*(-2)
    for i in range(nentries):
        im_file = sg_entries[i][1]['im_file']
        imdb_idx = np.where(im_paths==im_file)[0]
        rpn_idx = np.where(rpn_to_imdb==imdb_idx)[0][0]
        cases[i] = prop['im_to_case'][rpn_idx]
        outliers[i] = prop['im_to_outlier'][rpn_idx]
    return (cases, outliers)

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

parser = argparse.ArgumentParser(description='evaluate the accuracy of a passed experiment in multi label setting')
parser.add_argument('--dump_file')
parser.add_argument('--dict_file')
parser.add_argument('--prop')
parser.add_argument('--imdb')
args = parser.parse_args()

# loeading the data
dump_file = args.dump_file
dict_file = args.dict_file
label_dict =  json.load(open(dict_file)) 
sg_entries = pickle.load(open(dump_file))
prop = h5py.File(args.prop)
imdb = h5py.File(args.imdb) 
opposite_labels = get_opposite_labels(label_dict['predicate_to_idx'])

# shaping the predicted and gt relations arrays 
n_rel_cat = len(label_dict['idx_to_predicate']) + 1
(pred_rels, gt_rels) = get_rels(sg_entries, opposite_labels)
print 'accuracy for the relations:'
m_rel = np.mean(pred_rels[:,0]==gt_rels[:,3])
print m_rel

(pred_objs, gt_objs) = get_objs(sg_entries)
m_objs = np.mean(pred_objs[:,1]==gt_objs[:,2])
print 'accuracy of the object classification'
print m_objs

cases, outliers = get_cases(imdb, sg_entries, prop)
case_rels, outlier_rels = case_for_rel(gt_rels,cases, outliers)

print 'relations accuracy for 0 case'
print np.mean(pred_rels[case_rels==0]==gt_rels[np.nonzero(case_rels==0)[0],3])

print 'relation accuracy for 1 case not outlier'
idx = np.nonzero(np.logical_and(case_rels==1,  outlier_rels==-1))[0]
print np.mean( pred_rels[idx].transpose() == gt_rels[idx,3])

print 'relation accuracy for 1 case outlier'
idx = np.nonzero(np.logical_and(case_rels==1,  outlier_rels!=-1))[0]
print np.mean( pred_rels[idx].transpose() == gt_rels[idx,3])

print 'relation accuracy for 2 case not outlier'
idx = np.nonzero(np.logical_and(case_rels==2,  outlier_rels==-1))[0]
print np.mean( pred_rels[idx].transpose() == gt_rels[idx,3])

print 'relation accuracy for 2 case outlier'
idx = np.nonzero(np.logical_and(case_rels==2,  outlier_rels!=-1))[0]
print np.mean( pred_rels[idx].transpose() == gt_rels[idx,3])


case_objs, outlier_objs = case_for_obj(gt_objs, cases, outliers)
print 'relations objs for 0 case'
idx = np.nonzero(case_objs==0)[0]
print np.mean(pred_objs[idx,1]==gt_objs[idx,2])

print 'relation accuracy for 1 case not outlier'
idx = np.nonzero(np.logical_and(case_objs==1,  outlier_objs==-1))[0]
print np.mean( pred_objs[idx,1].transpose() == gt_objs[idx,2])

print 'relation accuracy for 1 case outlier'
idx = np.nonzero(np.logical_and(case_objs==1,  outlier_objs!=-1))[0]
print np.mean( pred_objs[idx,1].transpose() == gt_objs[idx,2])

print 'relation accuracy for 2 case not outlier'
idx = np.nonzero(np.logical_and(case_objs==2,  outlier_objs==-1))[0]
print np.mean( pred_objs[idx,1].transpose() == gt_objs[idx,2])

print 'relation accuracy for 2 case outlier'
idx = np.nonzero(np.logical_and(case_objs==2,  outlier_objs!=-1))[0]
print np.mean( pred_objs[idx,1].transpose() == gt_objs[idx,2])
