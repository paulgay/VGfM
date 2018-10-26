# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

"""Factory method for easily getting networks by name."""

__sets = {}

from models import *

__sets['vrd'] = vrd # VRD baseline
__sets['dual_graph_vrd_avgpool'] = dual_graph_vrd_avgpool  # avg pooling baseline
__sets['dual_graph_vrd_maxpool'] = dual_graph_vrd_maxpool  # max pooling baseline
__sets['dual_graph_vrd_final'] = dual_graph_vrd_final  # final model
__sets['dual_graph_vrd_3d'] = dual_graph_vrd_3d # video model
__sets['dual_graph_simple'] = dual_graph_simple
__sets['dual_graph_vrd_geo_only'] = dual_graph_vrd_geo_only # video model without using deep features
__sets['dual_graph_vrd_fus'] = dual_graph_vrd_fus 
__sets['dual_graph_vrd_fus_app'] = dual_graph_vrd_fus_app
__sets['dual_graph_vrd_geo_stand'] = dual_graph_vrd_geo_stand 
__sets['fus_early_multi'] = fus_early_multi
__sets['dual_graph_vrd_2d'] = dual_graph_vrd_2d
__sets['fus_2d3d'] = fus_2d3d
__sets['dual_graph_vrd_3d2'] = dual_graph_vrd_3d2
__sets['mono_multi'] = mono_multi
__sets['dual_3d_stopped'] = dual_3d_stopped
__sets['fus_simple'] = fus_simple 
__sets['dual_graph_vrd_label'] = dual_graph_vrd_label
def get_network(name):
    """Get a network by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown network: {}'.format(name))
    return __sets[name]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
