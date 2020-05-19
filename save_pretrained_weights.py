import config
import tensorflow as tf
from caps_network import Caps3d

capsnet = Caps3d()
with tf.Session(graph=capsnet.graph, config=config.gpu_config) as sess:
    tf.global_variables_initializer().run()
    capsnet.load(sess, config.network_save_dir)
    save_file_name = config.network_save_dir + 'pretrained_capsnet_83.ckpt'
    capsnet.save(sess, save_file_name, ep=100) 