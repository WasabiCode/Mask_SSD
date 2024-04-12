# Mask_SSD
Novel approach for object detection. Segmentation mask and Bounding boxes in one forward pass


In the field of computer vision, the primary techniques for determining object location in digital images are image segmentation and localization using bounding boxes. The fully convolutional U-Net, introduced in 2015, is one of the leading architectures for semantic segmentation, where each pixel of an image is assigned a specific class. This network has a two-part structure: an encoding part that extracts features from the image, allowing integration of pre-trained models from the ImageNet database, such as VGG16 or ResNet, and a decoding part that accurately reconstructs the segmented image. For object localization, the Single Shot Multibox Detector (SSD) architecture is also using feature extraction mechanisms from the image, followed by additional localization a classification layers.


Mask-SSD model integrates the two above mentioned methods and achieves up to five times faster processing compared to using them separately. The Mask-SSD model primarily aims to optimize speed and efficiency, making it suitable for applications that require fast image processing, such as autonomous driving. This is achieved by jointly leveraging the feature extractor of both original models, resulting in significant speed improvements. The model is being trained on the VOC2012 dataset, which contains approximately 1400 images, using the powerful Nvidia V100 GPU.

# TO DO
improve models accuracy
