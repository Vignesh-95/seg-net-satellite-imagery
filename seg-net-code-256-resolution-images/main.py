import tensorflow as tf
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', '/home/cs/students/u15031625/256_SegNet/Logs/model.ckpt-49999', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "8", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "./Logs", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "./Index/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "./Index/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "./Index/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "50000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "256", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "256", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "6", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)

def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("Potsdam testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("Potsdam Image dir: %s"%FLAGS.image_dir)
        print("Potsdam Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("Potsdam Image dir: %s"%FLAGS.image_dir)
        print("Potsdam Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)


def main(args):
    checkArgs()
    if FLAGS.testing:
        model.test(FLAGS)
    elif FLAGS.finetune:
        model.training(FLAGS, is_finetune=True)
    else:
        model.training(FLAGS, is_finetune=False)

if __name__ == '__main__':
  tf.app.run()
