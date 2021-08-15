import time

from AE import use_existing_markers
from FlatAE import FlatAutoEncoder
from utils.data import *
from utils.flags import FLAGS
from train import test

from tensorflow.core.protobuf import saver_pb2

class DataInfo(object):
    """Information about the datasets

     Will be passed to the FlatAe for creating corresponding variables in the graph
    """

    def __init__(self, data_sigma, train_shape, eval_shape, max_val):
        """DataInfo initializer

        Args:
          data_sigma:   variance in the dataset
          train_shape:  dimensionality of the train dataset
          eval_shape:   dimensionality of the evaluation dataset
        """
        self._data_sigma = data_sigma
        self._train_shape = train_shape
        self._eval_shape = eval_shape
        self._max_val = max_val

def run(input_seq_file_name):
    with tf.Graph().as_default():
        dropout = FLAGS.dropout # keep probability value
        variance = FLAGS.variance_of_noise
        #learning_rate = FLAGS.learning_rate
        batch_size = FLAGS.batch_size
        num_hidden = FLAGS.num_hidden_layers
        tf.set_random_seed(FLAGS.seed)
        ae_hidden_shapes = [FLAGS.network_width for j in range(num_hidden)]

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Read the data
        data, max_val, mean_pose = read_datasets_from_binary()
        data_info = DataInfo(data.train.sigma, data.train._sequences.shape,
                                     data.test._sequences.shape, max_val)
        # Pad max values and the mean pose, if needed
        if FLAGS.amount_of_frames_as_input > 1:
            max_val = np.tile(max_val, FLAGS.amount_of_frames_as_input)
            mean_pose = np.tile(mean_pose, FLAGS.amount_of_frames_as_input)

        ae_shape = [FLAGS.frame_size * FLAGS.amount_of_frames_as_input] + ae_hidden_shapes + [
                    FLAGS.frame_size * FLAGS.amount_of_frames_as_input]
        # create model
        ae = FlatAutoEncoder(ae_shape, sess, batch_size, variance, data_info)
        sess.run(tf.local_variables_initializer())

        with tf.variable_scope("test"):

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Create a saver
            saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

            # restore model from checkpoint
            chkpt_file = FLAGS.chkpt_dir + '/chkpt-' + str(FLAGS.chkpt_num)
            saver.restore(sess, chkpt_file)
            print("Model restored from the file " + str(chkpt_file) + '.')


            #print("---------TESTING------------")
            ##rmse = test(ae, FLAGS.data_dir + '/test_seq/boxing.binary', max_val, mean_pose, True)
            ##print("\nOur RMSE for boxing is : ", rmse)
            #rmse = test(ae, FLAGS.real_data_dir + '/shakehands.binary', max_val, mean_pose, True)
            #print("\nOur RMSE for shake hands is : ", rmse)

            #                    GET THE DATA

            # get input sequence
            #print('\nRead a test sequence from the file',input_seq_file_name,'...')
            original_input = read_test_seq_from_binary(input_seq_file_name)
            # Create an empty array for an output
            output_batches = np.array([])

            output_batch, mask = sess.run([ae._valid_output, ae._mask],
                                              feed_dict={ae._valid_input_: batches[batch_numb],
                                                         ae._mask: ae._mask_generator.eval(session=sess)})

if __name__ == '__main__':
    write_test_seq_in_binary(FLAGS.real_data_dir + '/Trial002-Reconstructed-SelectedFrame.c3d',
                        FLAGS.real_data_dir + '/shakehands.binary')
    write_test_seq_in_binary(FLAGS.real_data_dir + '/14_01.c3d',
                        FLAGS.real_data_dir + '/boxing_2.binary')

