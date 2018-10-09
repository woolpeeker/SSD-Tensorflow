
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import tf_extended as tfe
from .tf_mobilenet.mobilenet_v2 import mobilenet as tf_mobilenet
from nets import ssd_common
from nets import custom_layers

MobilenetParams = namedtuple('MobilenetParams', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class Mobilenet:
    default_params = MobilenetParams(
        img_shape=(512, 512),
        num_classes=2,
        no_annotation_label=2,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, MobilenetParams):
            self.params = params
        else:
            raise Exception('Wrong params type')

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return tfe.anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope='bboxes_encode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def mobilenet_net(self, inputs, prediction_fn=slim.softmax, is_training=True, reuse=None,):
        num_classes=self.params.num_classes
        feat_layers=self.params.feat_shapes
        anchor_sizes=self.params.anchor_sizes
        anchor_ratios=self.params.anchor_ratios

        end_points=dict()
        net=tf_mobilenet(inputs,depth_multiplier=1.4)
        scopes=['expanded_conv_'+str(i) for i in range(1,17)]
        scopes.insert(0,'expanded_conv')
        for scope in scopes:
            end_point_name='/'.join(['MobilenetV2',scope,'output:0'])
            end_point = tf.get_default_graph().get_tensor_by_name(end_point_name)
            end_points[scope] = end_point

        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],)
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
        return predictions, localisations, logits, end_points

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='losses'):
        """Define the network losses.
        """
        with tf.name_scope(scope, 'losses'):
            l_cross_pos = []
            l_cross_neg = []
            l_loc = []
            for i in range(len(logits)):
                dtype = logits[i].dtype
                with tf.name_scope('block_%i' % i):
                    # Determine weights Tensor.
                    pmask = gscores[i] > match_threshold
                    fpmask = tf.cast(pmask, dtype)
                    n_positives = tf.reduce_sum(fpmask)

                    # Select some random negative entries.
                    # n_entries = np.prod(gclasses[i].get_shape().as_list())
                    # r_positive = n_positives / n_entries
                    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                    # Negative mask.
                    no_classes = tf.cast(pmask, tf.int32)
                    predictions = slim.softmax(logits[i])
                    nmask = tf.logical_and(tf.logical_not(pmask),
                                           gscores[i] > -0.5)
                    fnmask = tf.cast(nmask, dtype)
                    nvalues = tf.where(nmask,
                                       predictions[:, :, :, :, 0],
                                       1. - fnmask)
                    nvalues_flat = tf.reshape(nvalues, [-1])
                    # Number of negative entries to select.
                    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                    n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                    n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                    max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                    n_neg = tf.minimum(n_neg, max_neg_entries)

                    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                    minval = val[-1]
                    # Final negative mask.
                    nmask = tf.logical_and(nmask, -nvalues > minval)
                    fnmask = tf.cast(nmask, dtype)

                    # Add cross-entropy loss.
                    with tf.name_scope('cross_entropy_pos'):
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=gclasses[i])
                        loss = tf.losses.compute_weighted_loss(loss, fpmask)
                        l_cross_pos.append(loss)

                    with tf.name_scope('cross_entropy_neg'):
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=no_classes)
                        loss = tf.losses.compute_weighted_loss(loss, fnmask)
                        l_cross_neg.append(loss)

                    # Add localization loss: smooth L1, L2, ...
                    with tf.name_scope('localization'):
                        # Weights Tensor: positive mask + random negative.
                        weights = tf.expand_dims(alpha * fpmask, axis=-1)
                        loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                        loss = tf.losses.compute_weighted_loss(loss, weights)
                        l_loc.append(loss)

            # Additional total losses...
            with tf.name_scope('total'):
                total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
                total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
                total_loc = tf.add_n(l_loc, 'localization')

                # Add to EXTRA LOSSES TF.collection
                tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.add_to_collection('EXTRA_LOSSES', total_loc)



def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1]):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tfe.get_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tfe.get_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred