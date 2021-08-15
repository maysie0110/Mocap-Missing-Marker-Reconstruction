"""
This is the main file of this project.

It contains the training function as well as the testing routine.

This file uses all the other files, such as AE.py, FlatAE.py and files from the folder utils

If you encounter any problems/bugs/issues please contact me on github
or by emailing me at tarask@kth.se for any bug reports/questions/suggestions.
"""
from __future__ import division
from __future__ import print_function

import time

from AE import use_existing_markers
from FlatAE import FlatAutoEncoder
from utils.data import *
from utils.flags import FLAGS

from tensorflow.core.protobuf import saver_pb2

SKIP = FLAGS.skip_duration      # skip first few seconds - to let motion begin
NO_GAP = FLAGS.no_gap_duration  # give all the markers for the first second

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


def learning(data, max_val, learning_rate, batch_size, dropout):
    """ Training of the denoising autoencoder

    Returns:
      Autoencoder trained on a data provided by FLAGS from utils/flags.py
    """

    with tf.Graph().as_default():

        tf.set_random_seed(FLAGS.seed)

        start_time = time.time()

        # Read the flags
        variance = FLAGS.variance_of_noise
        num_hidden = FLAGS.num_hidden_layers
        ae_hidden_shapes = [FLAGS.network_width for j in range(num_hidden)]

        # Check if recurrency is set in the correct way
        if FLAGS.reccurent == False and FLAGS.chunk_length > 1:
            print("ERROR: Without recurrency chunk length should be 1!"
                  " Please, change flags accordingly")
            exit(1)

        # Check if the flags makes sence
        if dropout < 0 or variance < 0:
            print('ERROR! Have got negative values in the flags!')
            exit(1)

        # Get the information about the dataset
        data_info = DataInfo(data.train.sigma, data.train._sequences.shape,
                             data.test._sequences.shape, max_val)

        # Allow tensorflow to change device allocation when needed
        config = tf.ConfigProto(allow_soft_placement=True) #,log_device_placement=True)
        # Adjust configuration so that multiple executions are possible
        config.gpu_options.allow_growth = True

        # Start a session
        sess = tf.Session(config=config)

        # Create an autoencoder
        ae_shape = [FLAGS.frame_size * FLAGS.amount_of_frames_as_input] + ae_hidden_shapes + [
            FLAGS.frame_size * FLAGS.amount_of_frames_as_input]
        ae = FlatAutoEncoder(ae_shape, sess, batch_size, variance, data_info)
        print('\nFlat AE was created : ', ae_shape)

        # Initialize input_producer
        sess.run(tf.local_variables_initializer())

        with tf.variable_scope("Train"):

            ##############        DEFINE  Optimizer and training OPERATOR      ############

            # Define optimizers
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # Do gradient clipping
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(ae._loss, tvars), 1e12)
            train_op = optimizer.apply_gradients\
                (zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

            # Prepare for making a summary for TensorBoard
            train_error = tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
            eval_error = tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')
            tf.summary.scalar('Train_error', train_error)
            train_summary_op = tf.summary.merge_all()
            eval_summary_op = tf.summary.scalar('Validation_error', eval_error)

            summary_dir = FLAGS.summary_dir
            summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

            num_batches = int(data.train._num_sequences / batch_size)

            # Initialize the part of the graph with the input data
            sess.run(ae._train_data.initializer,
                     feed_dict={ae._train_data_initializer: data.train._sequences})
            sess.run(ae._valid_data.initializer,
                     feed_dict={ae._valid_data_initializer: data.test._sequences})

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Create a saver
            saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

            # restore model, if needed
            if FLAGS.restore:
                chkpt_file = FLAGS.chkpt_dir + '/chkpt-' + str(FLAGS.chkpt_num)
                saver.restore(sess, chkpt_file)
                print("Model restored from the file " + str(chkpt_file) + '.')

            # A few initialization for the early stopping
            delta = FLAGS.delta_for_early_stopping  # error tolerance for early stopping
            best_error = 10000
            num_valid_batches = int(data.test.num_sequences / batch_size)

            try:  # running enqueue threads.

                # Train the whole network jointly
                step = 0
                print('\nWe train on ', num_batches, ' batches with ', batch_size,
                      ' training examples in each for', FLAGS.training_epochs, ' epochs...')
                print("")
                print(" ______________ ______")
                print("|     Epoch    | RMSE |")
                print("|------------  |------|")

                while not coord.should_stop():

                    if FLAGS.continuos_gap:
                        loss_summary, loss_value = sess.run([train_op, ae._reconstruction_loss],
                                                            feed_dict={ae._mask: cont_gap_mask()})
                    else:
                        loss_summary, loss_value = sess.run\
                            ([train_op, ae._reconstruction_loss],
                             feed_dict={ae._mask: ae._mask_generator.eval(session=ae.session)})

                    train_error_ = loss_value

                    if step % num_batches == 0:
                        epoch = step * 1.0 / num_batches

                        train_summary = sess.run(train_summary_op,
                                                 feed_dict={train_error: np.sqrt(train_error_)})

                        # Print results of screen
                        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
                        percent_str = "({0:3.2f}".format(epoch * 100.0 / FLAGS.training_epochs)[:5]
                        error_str = "%) |{0:5.2f}".format(train_error_)[:10] + "|"
                        print(epoch_str, percent_str, error_str)

                        if epoch % 5 == 0:

                            ##rmse = test(ae, FLAGS.data_dir + '/../test_seq/basketball_2.binary', max_val, mean_pose)
                            #rmse = test(ae, FLAGS.data_dir + '/test_seq/basketball.binary', max_val, mean_pose)
                            #print("\nOur RMSE for basketball is : ", rmse)

                            #rmse = test(ae, FLAGS.data_dir + '/../test_seq/boxing.binary', max_val, mean_pose)
                            rmse = test(ae, FLAGS.data_dir + '/test_seq/boxing.binary', max_val, mean_pose)
                            print("\nOur RMSE for boxing is : ", rmse)

                            #rmse = test(ae, FLAGS.data_dir + '/../test_seq/salto.binary',
                            #            max_val, mean_pose)#, True)
                            #print("\nOur RMSE for the jump turn is : ", rmse)

                        if epoch > 0:
                            summary_writer.add_summary(train_summary, step)

                            # Evaluate on the validation sequences
                            error_sum = 0
                            for valid_batch in range(num_valid_batches):
                                curr_err = sess.run\
                                    ([ae._valid_loss],
                                     feed_dict={ae._mask: ae._mask_generator.eval(session=sess)})
                                error_sum += curr_err[0]
                            new_error = error_sum / (num_valid_batches)
                            eval_sum = sess.run(eval_summary_op,
                                                feed_dict={eval_error: np.sqrt(new_error)})
                            summary_writer.add_summary(eval_sum, step)

                            # Early stopping
                            if FLAGS.Early_stopping and epoch > 20:
                                if (new_error - best_error) / best_error > delta:
                                    print('After ' + str(step) +
                                          ' steps the training started over-fitting ')
                                    break
                                if new_error < best_error:
                                    best_error = new_error

                                    # Saver for the model
                                    save_path = saver.save(sess, FLAGS.chkpt_dir + '/chkpt',
                                                           global_step=step)

                            if epoch % 5 == 0:
                                # Save for the model
                                save_path = saver.save(sess, FLAGS.chkpt_dir + '/chkpt',
                                                       global_step=step)
                                print('Done training for %d epochs, %d steps.' %
                                      (FLAGS.training_epochs, step))
                                print("The model was saved in file: %s" % save_path)

                    step += 1

            except tf.errors.OutOfRangeError:
                if not FLAGS.Early_stopping:
                    # Save the model
                    save_path = saver.save(sess, FLAGS.chkpt_dir + '/chkpt',
                                           global_step=step)
                print('Done training for %d epochs, %d steps.' % (FLAGS.training_epochs, step))
                print("The final model was saved in file: %s" % save_path)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

        duration = (time.time() - start_time) / 60  # in minutes, instead of seconds

        print("The training was running for %.3f  min" % (duration))

        # Save the results
        f = open(FLAGS.results_file, 'a')
        f.write('\nFor the data with ' + str(FLAGS.duration_of_a_gap) + ' gap ! '
                + ' the test error is ' + str.format("{0:.5f}", np.sqrt(new_error)))
        f.close()

        return ae

def test(ae, input_seq_file_name, max_val, mean_pose, write_skels_to_files=False):
    """
    Test our system on a particular sequence

    Args:
     ae:                    trained AE
     input_seq_file_name:   address of the binary file with a test sequence
     max_val:               max values in the dataset (for the normalization)
     mean_pose:             mean values in the dataset (for the normalization)
     write_skels_to_files:  weather we write the sequnces into a file (for further visualization)

    Returns:
     rmse                 root squared mean error
    """
    with ae.session.graph.as_default() as sess:
        sess = ae.session
        chunking_stride = FLAGS.chunking_stride


        #                    GET THE DATA

        # get input sequnce
        #print('\nRead a test sequence from the file',input_seq_file_name,'...')
        original_input = read_test_seq_from_binary(input_seq_file_name)

        visualizing = False
        if visualizing:
            visualize(original_input)

        if FLAGS.plot_error:
            # cut only interesting part of a sequence
            original_input = original_input[SKIP:SKIP +NO_GAP+FLAGS.duration_of_a_gap+NO_GAP]

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

            #                    MAKE A SEQUENCE WITH MISSING MARKERS

            output_batches = np.array([])

            # Go over all batches one by one
            for batch_numb in range(numb_of_batches):

                mask = mask_batches[batch_numb]

                # Simulate missing markers
                new_result = np.multiply(batches[batch_numb], mask)

                output_batches = np.append(output_batches, [new_result], axis=0) if \
                    output_batches.size else np.array([new_result])

            # No postprocessing
            output_sequence = reshape_from_batch_to_sequence(output_batches)

            noisy = output_sequence.reshape(-1, output_sequence.shape[-1])

            visualize(noisy)

            save_motion(noisy, input_seq_file_name + '_noisy.csv')


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

        #              CALCULATE the error for our network
        new_size = np.fmin(reconstructed.shape[0], original_input.shape[0])
        error = (reconstructed[0:new_size] - original_input[0:new_size]) * ae.scaling_factor
        # take into account only missing markers
        total_rmse = np.sqrt(((error[error > 0.000000001]) ** 2).mean())

        if FLAGS.plot_error:

            if not FLAGS.continuos_gap:
                print("ERROR! If you need to plot an error - you should have a continuosly "
                      "missing markers. Change flags.py accordingly")
                print("For example: set flag 'continuos_gap' to True")
                exit(0)

            assert FLAGS.duration_of_a_gap < error.shape[0] * FLAGS.amount_of_frames_as_input


            # Calculate error for every frame
            better_error = np.zeros([FLAGS.duration_of_a_gap + NO_GAP])
            for i in range(int(FLAGS.duration_of_a_gap/FLAGS.amount_of_frames_as_input)):

                # Convert from many frames at a time - to just one frame at at time
                if not FLAGS.reccurent:
                    new_error = error[i + int(NO_GAP/FLAGS.amount_of_frames_as_input)].\
                        reshape(-1, FLAGS.frame_size)

                    for time in range(FLAGS.amount_of_frames_as_input):
                        this_frame_err = new_error[time]
                        rmse = np.sqrt(((this_frame_err[this_frame_err > 0.000000001]) ** 2).mean())

                        if rmse > 0:
                            better_error[i * FLAGS.amount_of_frames_as_input + time+ NO_GAP] = rmse

                else:
                    this_frame_err = error[i+ NO_GAP]
                    rmse = np.sqrt(((this_frame_err[this_frame_err > 0.000000001]) ** 2).mean())
                    if rmse > 0:
                        better_error[i+ NO_GAP] = rmse

            with open(FLAGS.contin_test_file, 'w') as file_handler:
                for item in better_error:
                    file_handler.write("{}\n".format(item))
                file_handler.close()

        return total_rmse

def reshape_from_batch_to_sequence(input_batch):
    '''
    Reshape batch of overlapping sequences into 1 sequence

    Args:
         input_batch: batch of overlapping sequences
    Return:
         flat_sequence: one sequence with the same values

    '''

    # Get the data from the Flags
    chunking_stride = FLAGS.chunking_stride
    if FLAGS.reccurent:
        sequence_length = FLAGS.chunk_length
    else:
        sequence_length = FLAGS.amount_of_frames_as_input

    # Reshape batches
    input_chunks = input_batch.reshape(-1, input_batch.shape[2], input_batch.shape[3])
    numb_of_chunks = input_chunks.shape[0]

    if FLAGS.reccurent:
        # Map from overlapping windows to non-overlaping
        # Take first chunk as a whole and the last part of each other chunk

        input_non_overlaping = input_chunks[0]
        for i in range(1, numb_of_chunks, 1):

            input_non_overlaping = np.concatenate(
                (input_non_overlaping,
                 input_chunks[i][sequence_length - chunking_stride: sequence_length][:]), axis=0)

        input_non_overlaping = np.array(input_non_overlaping)

    else:
        input_non_overlaping = input_chunks.reshape(input_chunks.shape[0], 1,
                                                    sequence_length * FLAGS.frame_size)

    # Flaten it into a sequence
    flat_sequence = input_non_overlaping.reshape(-1, input_non_overlaping.shape[-1])

    return flat_sequence


def convert_back_to_3d_coords(sequence, max_val, mean_pose):
    '''
    Convert back from the normalized values between -1 and 1 to original 3d coordinates
    and unroll them into the sequence

    Args:
        sequence: sequence of the normalized values
        max_val: maximal value in the dataset
        mean_pose: mean value in the dataset

    Return:
        3d coordinates corresponding to the batch
    '''

    # Convert it back from the [-1,1] to original values
    reconstructed = np.multiply(sequence, max_val[np.newaxis, :] + 1e-15)

    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis, :]

    # Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])

    return reconstructed


def get_the_data():
    data, max_val, mean_pose = read_datasets_from_binary()

    # Check, if we have enough data
    if FLAGS.batch_size > data.train._num_chunks:
        print('ERROR! Cannot have less train sequences than a batch size!')
        exit(1)
    if FLAGS.batch_size > data.test._num_chunks:
        print('ERROR! Cannot have less test sequences than a batch size!')
        exit(1)

    return data, max_val, mean_pose

def cont_gap_mask(length=0, gap_begins=0, test=False):

    if not test:
        mask_size = [FLAGS.batch_size, FLAGS.chunk_length,
                     int(FLAGS.frame_size * FLAGS.amount_of_frames_as_input)]
        length = FLAGS.chunk_length
    else:
        mask_size = [1, length, int(FLAGS.frame_size * FLAGS.amount_of_frames_as_input)]

    mask = np.ones(mask_size)
    probabilities = [1.0 / (41) for marker in range(41)]

    for batch in range(mask_size[0]):

        start_fr = int(gap_begins/FLAGS.amount_of_frames_as_input)

        if test:
            if FLAGS.duration_of_a_gap:
                gap_length = FLAGS.duration_of_a_gap
            else:
                gap_length = int(length/FLAGS.amount_of_frames_as_input)
        else:
            gap_length = length

        time_fr = start_fr
        while time_fr < gap_length+start_fr:

            # choose random amount of time frames for a gap
            if FLAGS.duration_of_a_gap:
                gap_duration = FLAGS.duration_of_a_gap
            else:
                # between 0.1s and 1s (frame rate 60 fps)
                gap_duration = int(np.random.normal(120, 20))

            # choose random markers for the gap
            if FLAGS.amount_of_missing_markers < 21:
                random_markers = np.random.choice(41, FLAGS.amount_of_missing_markers,
                                                  replace=False, p=probabilities)
            else:
                random_markers = np.random.choice(41, FLAGS.amount_of_missing_markers,
                                                  replace=False)

            for gap_time in range(gap_duration):

                for muptipl_inputs in range(FLAGS.amount_of_frames_as_input):

                    for marker in random_markers:

                        mask[batch][time_fr][marker + 123*muptipl_inputs] = 0
                        mask[batch][time_fr][marker + 41+ 123*muptipl_inputs] = 0
                        mask[batch][time_fr][marker + 82+ 123*muptipl_inputs] = 0

                time_fr += 1
                if time_fr >= gap_length+start_fr:
                    break

            # Make sure not to use the same markers twice in a raw
            p = 1.0 / (41 - FLAGS.amount_of_missing_markers)
            probabilities = [0 if marker in random_markers else p for marker in range(41)]

    return mask


def save_motion(motion, file_name):
    """
    Save the motion into a csv file
    :param motion:     sequence of the motion 3d coordinates
    :param file_name:  file to write the motion into
    :return:           nothing
    """

    with open(file_name, 'w') as fp:

        if not FLAGS.reccurent:
            # Reshape input - to have just one frame at a time
            to_output = motion.reshape(-1, FLAGS.frame_size)
        else:
            to_output = motion

        np.savetxt(fp, to_output, delimiter=",")
        print("Motion was written to " + file_name)

if __name__ == '__main__':

    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    dropout = FLAGS.dropout  # keep probability value

    # Read the data
    data, max_val, mean_pose = read_datasets_from_binary()

    # Check, if we have enough data
    if FLAGS.batch_size > data.train.num_sequences:
        print('ERROR! Cannot have less train sequences than a batch size!')
        exit(1)
    if FLAGS.batch_size > data.test.num_sequences:
        print('ERROR! Cannot have less test sequences than a batch size!')
        exit(1)

    # Pad max values and the mean pose, if neeeded
    if FLAGS.amount_of_frames_as_input > 1:
        max_val = np.tile(max_val, FLAGS.amount_of_frames_as_input)
        mean_pose = np.tile(mean_pose, FLAGS.amount_of_frames_as_input)

    # Train the network
    ae = learning(data, max_val, learning_rate, batch_size, dropout)

    # TEST it
    #rmse = test(ae, FLAGS.data_dir + '/../test_seq/boxing.binary', max_val, mean_pose, True)
    rmse = test(ae, FLAGS.data_dir + '/test_seq/boxing.binary', max_val, mean_pose, True)
    print("\nOur RMSE for boxing is : ", rmse)

    ##rmse = test(ae, FLAGS.data_dir + '/../test_seq/basketball_2.binary', max_val, mean_pose, True)
    #rmse = test(ae, FLAGS.data_dir + '/test_seq/basketball.binary', max_val, mean_pose, True)
    #print("\nOur RMSE for basketball is : ", rmse)

    # Close Tf session
    ae.session.close()
