import numpy as np

def create_graph_data(num_roi, num_rel, relations):
    """
    compute graph structure from relations
    """

    rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool) # relations[k]=(i,j); rel_mask[i,k]=True; rel_mask[j,k]=True; roi_rel_inds[i,j]=k
    roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    for i, rel in enumerate(relations):
        rel_mask[rel[0], i] = True
        rel_mask[rel[1], i] = True
        roi_rel_inds[rel[0], rel[1]] = i

    rel_mask_inds = [] # indicate in which relations this segment is invovled and at the end is appended  the total number of relations
    rel_segment_inds = [] # indicate which segment this relation are about. 
    # rel_mask_inds = [ 2 4 8 1 5 8] and rel_segment_inds [0 0 0 1 1 1] indicate that segment 0 is in the relations 2 and 4. i is the segments inds and goes 1..num_roi
    for i, mask in enumerate(rel_mask):
        mask_inds = np.where(mask)[0].tolist() + [num_rel]
        segment_inds = [i for _ in mask_inds]
        rel_mask_inds += mask_inds
        rel_segment_inds += segment_inds

    # compute relation pair inds
    rel_pair_mask_inds = []  # for each roi i, contains the set of edges which are going in and out.
    rel_pair_segment_inds = []  # vector used to know which segment these edges are about, same system as in the previous loop 
    for i in xrange(num_roi):
        mask_inds = []
        for j in xrange(num_roi):
            out_inds = roi_rel_inds[i,j]
            in_inds = roi_rel_inds[j,i]
            if out_inds >= 0 and in_inds >= 0:
                out_inds = out_inds if out_inds >=0 else num_rel
                in_inds = in_inds if in_inds >=0 else num_rel
                mask_inds.append([out_inds, in_inds])

        mask_inds.append([num_rel, num_rel]) # pad with dummy edge ind
        rel_pair_mask_inds += mask_inds
        rel_pair_segment_inds += [i for _ in mask_inds]

    # sanity check
    for i, inds in enumerate(rel_pair_mask_inds):
        if inds[0] < num_rel:
            assert(relations[inds[0]][0] == rel_pair_segment_inds[i])
        if inds[1] < num_rel:
            assert(relations[inds[1]][1] == rel_pair_segment_inds[i])

    output_dict = {
        'rel_mask_inds': np.array(rel_mask_inds).astype(np.int32),
        'rel_segment_inds': np.array(rel_segment_inds).astype(np.int32),
        'rel_pair_segment_inds': np.array(rel_pair_segment_inds).astype(np.int32), # edge_pair_segment_inds
        'rel_pair_mask_inds': np.array(rel_pair_mask_inds).astype(np.int32),
        'num_roi': num_roi,
        'num_rel': num_rel
    }
    return output_dict


def create_graph_data_fus(num_roi, num_rel, relations, oids):
    """
    compute graph structure from relations
    relations [[o11, o12], .... [oi1, oi2]...
    oids[o11] gives the sequence level id of object o11: if oids[o11]==oids[o12] then o11 and o12 have been matched by the tracker to the same object.
    """

    rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool) # relations[k]=(i,j); rel_mask[i,k]=True; rel_mask[j,k]=True; roi_rel_inds[i,j]=k
    roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    for i, rel in enumerate(relations):
        rel_mask[rel[0], i] = True
        rel_mask[rel[1], i] = True
        roi_rel_inds[rel[0], rel[1]] = i

    rel_mask_inds = [] # indicate in which relations this segment is invovled and at the end is appended  the total number of relations
    rel_segment_inds = [] # indicate which segment this relation are about. 
    # rel_mask_inds = [ 2 4 8 1 5 8] and rel_segment_inds [0 0 0 1 1 1] indicate that segment 0 is in the relations 2 and 4. i is the segments inds and goes 1..num_roi
    for i, mask in enumerate(rel_mask):
        mask_inds = np.where(mask)[0].tolist() + [num_rel]
        segment_inds = [i for _ in mask_inds]
        rel_mask_inds += mask_inds
        rel_segment_inds += segment_inds

    # compute relation pair inds
    rel_pair_mask_inds = []  # for each roi i, contains the set of edges (the relations indices) which are going in and out.
    rel_pair_segment_inds = []  # vector used to know which segment these edges are about, same system as in the previous loop 
    for i in xrange(num_roi):
        mask_inds = []
        for j in xrange(num_roi):# collecting the nodes whiche are linked to i
            out_inds = roi_rel_inds[i,j] #out_inds is the index of the relation ie the relation node
            in_inds = roi_rel_inds[j,i]
            if out_inds >= 0 and in_inds >= 0:
                out_inds = out_inds if out_inds >=0 else num_rel
                in_inds = in_inds if in_inds >=0 else num_rel
                mask_inds.append([out_inds, in_inds]) # add the relation, so in the end, mask_inds will contain the relations where the node i is present

        mask_inds.append([num_rel, num_rel]) # pad with dummy edge ind
        rel_pair_mask_inds += mask_inds
        rel_pair_segment_inds += [i for _ in mask_inds] # creating [i]* len(mask_inds)

    obj_fus_mask_inds = [] #collect all the nodes where object i appears ie its apparitions in the other frames
    obj_fus_segment_inds = [] # same system as in the previous two loops, the roi obj_fus_mask_inds[i] corresponds to the object vid_segment_inds[i]
    for i in xrange(num_roi):
        oid = oids[i]
        mask_inds = list(np.where(oid==oids)[0])  # here I collect all the nodes where object i appears ie its apparition in the other frames
        mask_inds.remove(i) # avoiding that the node sends a message to itself
        obj_fus_mask_inds += mask_inds
        obj_fus_segment_inds += [i for _ in mask_inds] #  

    rel_fus_mask_inds = [] #collect all the nodes where rel i appears ie its apparitions in the other frames
    rel_fus_segment_inds = [] # same system as in the previous two loops, the roi obj_fus_mask_inds[i] corresponds to the object vid_segment_inds[i]
    rels_oids = make_rel_with_oid(relations, oids)
    for i in xrange(num_rel):
        oid1, oid2 = rels_oids[i]
        mask_inds = [ j for (j,(o1,o2)) in enumerate(rels_oids) if o1 == oid1 and o2 == oid2 ] #list(np.where(oid==oids)[0])  # here I collect all the nodes where object i appears ie its apparition in the other frames
        mask_inds.remove(i)
        rel_fus_mask_inds += mask_inds
        rel_fus_segment_inds += [i for _ in mask_inds] #  

    # sanity check
    for i, inds in enumerate(rel_pair_mask_inds):
        if inds[0] < num_rel:
            assert(relations[inds[0]][0] == rel_pair_segment_inds[i])
        if inds[1] < num_rel:
            assert(relations[inds[1]][1] == rel_pair_segment_inds[i])

    output_dict = {
        'rel_mask_inds': np.array(rel_mask_inds).astype(np.int32),
        'rel_segment_inds': np.array(rel_segment_inds).astype(np.int32),
        'rel_pair_segment_inds': np.array(rel_pair_segment_inds).astype(np.int32), # edge_pair_segment_inds
        'rel_pair_mask_inds': np.array(rel_pair_mask_inds).astype(np.int32),
        'obj_fus_segment_inds': np.array(obj_fus_mask_inds).astype(np.int32),
        'obj_fus_mask_inds': np.array(obj_fus_segment_inds).astype(np.int32),
        'rel_fus_segment_inds': np.array(rel_fus_mask_inds).astype(np.int32),
        'rel_fus_mask_inds': np.array(rel_fus_segment_inds).astype(np.int32),
        'num_roi': num_roi,
        'num_rel': num_rel
    }
    return output_dict

def make_rel_with_oid(relations, oids):
    rel_oids = []
    for (o1, o2) in relations:
        rel_oids.append((oids[o1],oids[o2]))
    return rel_oids


def get_rel_blob(feat2d, feat3d, rel_roidb, rels): #roidb[im_i]['rel_geo_2d'], roidb[im_i]['rel_geo_3d'], roidb[im_i]['gt_relations'], rels):
    no = np.max(rels[:,:2])+1
    o2do = {}
    for i in range(no):
       if i < no/2:
	   o2do[i] = i
       else:
	   o2do[i] = i - no/2
    np.max(rels[:,:2])
    f2d = np.zeros((rels.shape[0], feat2d.shape[1]))
    f3d = np.zeros((rels.shape[0], feat3d.shape[1]))
    for (i,(o1,o2,p)) in enumerate(rels):
	oo1,oo2 = o2do[o1], o2do[o2]
        if p!=0:
	    ii = np.where(np.logical_and(rel_roidb[:,0]==oo1, rel_roidb[:,1]==oo2))[0][0]
	    f2d[i] = feat2d[ii]
	    f3d[i] = feat3d[ii]
    return (f2d,f3d)


def create_graph_data_vid(num_roi, num_rel, relations, oids):
    """
    compute graph structure from relations
    """

    rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool) # relations[k]=(i,j); rel_mask[i,k]=True; rel_mask[j,k]=True; roi_rel_inds[i,j]=k
    roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    for i, rel in enumerate(relations):
        rel_mask[rel[0], i] = True
        rel_mask[rel[1], i] = True
        roi_rel_inds[rel[0], rel[1]] = i

    rel_mask_inds = [] # indicate in which relations this segment is invovled and at the end is appended  the total number of relations
    rel_segment_inds = [] # indicate which segment this relation are about. 
    # rel_mask_inds = [ 2 4 8 1 5 8] and rel_segment_inds [0 0 0 1 1 1] indicate that segment 0 is in the relations 2 and 4. i is the segments inds and goes 1..num_roi
    for i, mask in enumerate(rel_mask):
        mask_inds = np.where(mask)[0].tolist() + [num_rel]
        segment_inds = [i for _ in mask_inds]
        rel_mask_inds += mask_inds
        rel_segment_inds += segment_inds

    # compute relation pair inds
    rel_pair_mask_inds = []  # for each roi i, contains the set of edges which are going in and out.
    rel_pair_segment_inds = []  # vector used to know which segment these edges are about, same system as in the previous loop 
    for i in xrange(num_roi):
        mask_inds = []
        for j in xrange(num_roi):# collecting the nodes whiche are linked to i
            out_inds = roi_rel_inds[i,j] #out_inds is the index of the relation ie the relation node
            in_inds = roi_rel_inds[j,i]
            if out_inds >= 0 and in_inds >= 0:
                out_inds = out_inds if out_inds >=0 else num_rel
                in_inds = in_inds if in_inds >=0 else num_rel
                mask_inds.append([out_inds, in_inds]) # add the relation, so in the end, mask_inds will contain the relations where the node i is present

        mask_inds.append([num_rel, num_rel]) # pad with dummy edge ind
        rel_pair_mask_inds += mask_inds
        rel_pair_segment_inds += [i for _ in mask_inds]

    vid_mask_inds = []
    vid_segment_inds = []
    for i in xrange(num_roi):
        oid = oids[i]
        mask_inds = list(np.where(oid==oids)[0])  # here I collect all the nodes where object i appears ie its apparition in the other frames
        mask_inds.remove(i)
        vid_mask_inds += mask_inds
        vid_segment_inds += [i for _ in mask_inds] # mask_inds is size 


    # sanity check
    for i, inds in enumerate(rel_pair_mask_inds):
        if inds[0] < num_rel:
            assert(relations[inds[0]][0] == rel_pair_segment_inds[i])
        if inds[1] < num_rel:
            assert(relations[inds[1]][1] == rel_pair_segment_inds[i])

    output_dict = {
        'rel_mask_inds': np.array(rel_mask_inds).astype(np.int32),
        'rel_segment_inds': np.array(rel_segment_inds).astype(np.int32),
        'rel_pair_segment_inds': np.array(rel_pair_segment_inds).astype(np.int32), # edge_pair_segment_inds
        'rel_pair_mask_inds': np.array(rel_pair_mask_inds).astype(np.int32),
        'vid_segment_inds': np.array(vid_mask_inds).astype(np.int32),
        'vid_mask_inds': np.array(vid_segment_inds).astype(np.int32),
        'num_roi': num_roi,
        'num_rel': num_rel
    }
    return output_dict


def compute_rel_rois(num_rel, rois, relations):
    """
    union subject boxes and object boxes given a set of rois and relations
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert(sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                    np.minimum(sub_roi[1], obj_roi[1]),
                    np.maximum(sub_roi[2], obj_roi[2]),
                    np.maximum(sub_roi[3], obj_roi[3])]
        rel_rois[i, 1:] = union_roi

    return rel_rois
