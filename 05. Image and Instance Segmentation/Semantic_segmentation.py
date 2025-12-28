import pixellib
from pixellib import semantic

segment_image = semantic()
segment_image.load_pascalvoc_model('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
segment_image.segmentAsPascalVoc('01.jpg', output_image_name='semantic.jpg')