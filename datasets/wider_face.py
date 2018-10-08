import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

DATASET_SIZE = {
    'wider_face_val': 3226,
    'wider_face_test': 5011,
    'wider_face_train': 12880,
    'wider_demo_train': 1
}

def get_wider_demo_train_num_samples():
    return DATASET_SIZE['wider_demo_train']

def get_wider_demo_train_data(is_training_data=True):
    data_sources = "datasets/widerface_demo/widerface_demo*.tfrecord"
    num_samples = DATASET_SIZE['wider_demo_train']

    return get_images_labels_bboxes(data_sources, num_samples, is_training_data)


ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}


def get_images_labels_bboxes(data_sources, num_samples, is_training_data):
    '''

    :param data_sources: .tfrecord files
    :param num_samples:  number of samples in all the files
    :param is_training_data:
    :return:
    '''
    batch_size=FLAGS.batch_size
    dataset = get_dataset_info(data_sources, num_samples)
    is_training_data = is_training_data
    if is_training_data:

        shuffle = True
        # make sure most samples can be fetched in one epoch
        num_readers = 4
    else:
        # make sure data is fetchd in sequence
        shuffle = False
        num_readers = 1

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=shuffle,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)

    # Get for SSD network: image, labels, bboxes.
    #[image, shape, format, filename, glabels, gbboxes, gdifficults]
    return provider.get(['image', 'shape', 'format', 'filename',
         'object/label',
         'object/bbox',
         'object/difficult'])



def get_dataset_info(data_sources, num_samples):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """

    # Allowing None in the signature so that dataset_factory can use the default.

    reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value='000000'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64)
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
        'format': slim.tfexample_decoder.Tensor('image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None

    return slim.dataset.Dataset(
        data_sources=data_sources,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=1,
        labels_to_names=labels_to_names)
