import time

from AE import use_existing_markers
from FlatAE import FlatAutoEncoder
from utils.data import *
from utils.flags import FLAGS
from train import *
from graph import animate

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

def run(input_seq_file_name,write_skels_to_files=False):

    with tf.Graph().as_default():
        dropout = FLAGS.dropout # keep probability value
        variance = FLAGS.variance_of_noise
        #learning_rate = FLAGS.learning_rate
        batch_size = FLAGS.batch_size
        num_hidden = FLAGS.num_hidden_layers
        tf.set_random_seed(FLAGS.seed)
        ae_hidden_shapes = [FLAGS.network_width for j in range(num_hidden)]

        # Create a session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
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


            print("---------TESTING on REAL DATA------------")
            ##rmse = test(ae, FLAGS.data_dir + '/test_seq/boxing.binary', max_val, mean_pose, True)
            ##print("\nOur RMSE for boxing is : ", rmse)
            #rmse = test(ae, FLAGS.real_data_dir + '/shakehands.binary', max_val, mean_pose, True)
            #print("\nOur RMSE for shake hands is : ", rmse)

            chunking_stride = FLAGS.chunking_stride
            #                    GET THE DATA

            # get input sequence
            print('\nRead a test sequence from the file',input_seq_file_name,'...')
            original_input = read_test_seq_from_binary(input_seq_file_name)


            visualizing = False
            if visualizing:
                visualize(original_input)

            # Get a mask with very long gaps
            long_mask = cont_gap_mask(original_input.shape[0], NO_GAP, test=True)
  
            if long_mask.shape[1] < ae.sequence_length:
                print("ERROR! Your gap is too short for your sequence length")
                exit(0)

            mask_chunks = np.array([long_mask[0, i:i + ae.sequence_length, :] for i in
                                    range(0, len(long_mask[0]) - ae.sequence_length + 1,
                                           chunking_stride)])

            # Pad with itself if it is too short
            if mask_chunks.shape[0] < ae.batch_size:
                mupliplication_factor = int(ae.batch_size / mask_chunks.shape[0]) + 1
                mask_chunks = np.tile(mask_chunks, (mupliplication_factor, 1, 1))

            # Batch those chunks
            mask_batches = np.array([mask_chunks[i:i + ae.batch_size, :] for i in
                                     range(0, len(mask_chunks) - ae.batch_size + 1, ae.batch_size)])

            if write_skels_to_files:

                # No Preprocessing!
                coords_normalized = original_input

                save_motion(original_input, input_seq_file_name + '_original.csv')
                animate(input_seq_file_name + '_original.csv',True)

                if coords_normalized.shape[0] < ae.sequence_length:
                    mupliplication_factor = int(ae.batch_size * ae.sequence_length
                                                / coords_normalized.shape[0]) + 1
                    # Pad the sequence with itself in order to fill the batch completely
                    coords_normalized = np.tile(coords_normalized, mupliplication_factor)
                    print("Test sequence was way to short!")

                # Split it into chunks
                seq_chunks = np.array([coords_normalized[i:i + ae.sequence_length, :] for i in
                                       range(0, len(original_input) - ae.sequence_length + 1,
                                              chunking_stride)])  # Split sequence into chunks

                original_size = seq_chunks.shape[0]

                if original_size < ae.batch_size:
                    mupliplication_factor = int(ae.batch_size / seq_chunks.shape[0]) + 1

                    # Pad the sequence with itself in order to fill the batch completely
                    seq_chunks = np.tile(seq_chunks, (mupliplication_factor, 1, 1))

                # Batch those chunks
                batches = np.array([seq_chunks[i:i + ae.batch_size, :] for i in
                                    range(0, len(seq_chunks) - ae.batch_size + 1, ae.batch_size)])

                numb_of_batches = batches.shape[0]

    #                    MAKE AN OUTPUT SEQUENCE

            # Preprocess...
            coords_minus_mean = original_input - mean_pose[np.newaxis, :]
            eps = 1e-15
            coords_normalized = np.divide(coords_minus_mean, max_val[np.newaxis, :] + eps)

            if coords_normalized.shape[0] < ae.sequence_length:
                mupliplication_factor = (ae.batch_size * ae.sequence_length /
                                         coords_normalized.shape[0]) + 1
                # Pad the sequence with itself in order to fill the batch completely
                coords_normalized = np.tile(coords_normalized, mupliplication_factor)
                print("Test sequence was way to short!")

            # Split it into chunks
            seq_chunks = np.array([coords_normalized[i:i + ae.sequence_length, :] for i in
                                   range(0, len(original_input) - ae.sequence_length + 1,
                                          chunking_stride)])  # Split sequence into chunks

            # Pad with itself if it is too short
            if seq_chunks.shape[0] < ae.batch_size:
                mupliplication_factor = int(ae.batch_size / seq_chunks.shape[0]) + 1
                # Pad the sequence with itself in order to fill the batch completely
                seq_chunks = np.tile(seq_chunks, (mupliplication_factor, 1, 1))

            # Batch those chunks
            batches = np.array([seq_chunks[i:i + ae.batch_size, :] for i in
                                range(0, len(seq_chunks) - ae.batch_size + 1, ae.batch_size)])

            numb_of_batches = batches.shape[0]

            # Create an empty array for an output
            output_batches = np.array([])

            # Go over all batches one by one
            for batch_numb in range(numb_of_batches):
                if FLAGS.continuos_gap:
                    output_batch, mask = sess.run([ae._valid_output, ae._mask],
                                                  feed_dict={ae._valid_input_: batches[batch_numb],
                                                             ae._mask: mask_batches[batch_numb]})
                else:
                    output_batch, mask = sess.run([ae._valid_output, ae._mask],
                                                  feed_dict={ae._valid_input_: batches[batch_numb],
                                                             ae._mask:
                                                                ae._mask_generator.eval(session=sess)})

                # Take known values into account
                new_result = use_existing_markers(batches[batch_numb], output_batch, mask,
                                                  FLAGS.defaul_value)

                output_batches = np.append(output_batches, [new_result], axis=0) if output_batches.size\
                    else np.array([new_result])

            # Postprocess...
            output_sequence = reshape_from_batch_to_sequence(output_batches)

            reconstructed = convert_back_to_3d_coords(output_sequence, max_val, mean_pose)

            if write_skels_to_files:
                visualize(reconstructed, original_input)
                save_motion(reconstructed, input_seq_file_name + '_our_result.csv')
                animate(input_seq_file_name + '_our_result.csv',True)

if __name__ == '__main__':
    #write_test_seq_in_binary(FLAGS.real_data_dir + '/Trial002-Reconstructed-SelectedFrame.c3d',
    #                    FLAGS.real_data_dir + '/shakehands.binary')
    #write_test_seq_in_binary(FLAGS.real_data_dir + '/14_01.c3d',
    #                    FLAGS.real_data_dir + '/boxing_2.binary')

    run(FLAGS.real_data_dir + '/convo-May.binary', True)