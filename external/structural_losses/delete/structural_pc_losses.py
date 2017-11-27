'''
Created on Nov 2, 2017

@author: optas
'''

import tensorflow as tf

losses_found = True

try:    
    from tf_nndistance import nn_distance
    from tf_approxmatch import approx_match, match_cost
except:
    losses_found = False
    print('External Losses (Chamfer-EMD) were not loaded.')


def losses():
    if losses_found:
        return nn_distance, approx_match, match_cost
    else:
        return None, None, None
