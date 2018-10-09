
import tensorflow as tf
import lzma, pickle
import re, os

#this function copy from cifar10-tutorial
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if grads:
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
        else:
            continue

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tenserboard_writer(path='tensorboard', fold='run'):
    assert os.path.exists(path)
    max_run=[0]
    for d in os.listdir(path):
        d=os.path.join(path,d)
        if os.path.isdir(d):
            match=re.search('[\\/]run(\d+)$',d)
            if match:
                max_run.append(int(match.group(1)))
    run_num=max(max_run)+1
    result_path=os.path.join(path,fold+str(run_num))
    return tf.summary.FileWriter(result_path),run_num