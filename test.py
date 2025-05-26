import csv
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import os
import time # For checkpoint directory finding
import datetime # For consistency, though not strictly needed for test.py

def main():
    print("Starting test.py script...")

    # --------------------------------------------------------------------------
    # 1. Constants & CharCNN Model Definition (Mirrors train.py)
    # --------------------------------------------------------------------------
    print("Step 1: Defining constants and CharCNN model...")

    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\\\"/\\\\|_@#$%^&*~`+ =<>()[]{}"
    FEATURE_LEN = 1014
    cdict = {c: i + 2 for i, c in enumerate(ALPHABET)}
    
    # num_classes_actual will be determined after loading test data labels
    
    class CharCNN(tf.keras.Model):
        def __init__(self, num_classes, sequence_max_length=FEATURE_LEN, 
                     num_quantized_chars=len(ALPHABET) + 2, embedding_size=16, 
                     l2_reg_lambda=0.001, dropout_rate=0.0): 
            super(CharCNN, self).__init__(name='CharCNN')
            self.num_classes = num_classes
            self.sequence_max_length = sequence_max_length
            self.num_quantized_chars = num_quantized_chars
            self.embedding_size = embedding_size
            self.l2_reg_lambda = l2_reg_lambda
            self.dropout_rate = dropout_rate

            self.embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.num_quantized_chars,
                output_dim=self.embedding_size,
                input_length=self.sequence_max_length,
                embeddings_initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
                name='embedding_W')

            self.conv0_layer = tf.keras.layers.Conv2D(
                filters=64, kernel_size=(3, self.embedding_size), strides=(1,1), 
                padding="SAME", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='conv0')
            self.bn0_layer = tf.keras.layers.BatchNormalization(name='bn0')

            # Convolutional Blocks Layers
            # To ensure layers are properly tracked, they are defined as attributes of the class
            self.cb1_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb1_conv1')
            self.cb1_bn1 = tf.keras.layers.BatchNormalization(name='cb1_bn1')
            self.cb1_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb1_conv2')
            self.cb1_bn2 = tf.keras.layers.BatchNormalization(name='cb1_bn2')
            self.cb1_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb1_pool')

            self.cb2_conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb2_conv1')
            self.cb2_bn1 = tf.keras.layers.BatchNormalization(name='cb2_bn1')
            self.cb2_conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb2_conv2')
            self.cb2_bn2 = tf.keras.layers.BatchNormalization(name='cb2_bn2')
            self.cb2_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb2_pool')

            self.cb3_conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb3_conv1')
            self.cb3_bn1 = tf.keras.layers.BatchNormalization(name='cb3_bn1')
            self.cb3_conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb3_conv2')
            self.cb3_bn2 = tf.keras.layers.BatchNormalization(name='cb3_bn2')
            self.cb3_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb3_pool')

            self.cb4_conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb4_conv1')
            self.cb4_bn1 = tf.keras.layers.BatchNormalization(name='cb4_bn1')
            self.cb4_conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb4_conv2')
            self.cb4_bn2 = tf.keras.layers.BatchNormalization(name='cb4_bn2')
            self.cb4_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb4_pool')
            
            regularizer = tf.keras.regularizers.L2(self.l2_reg_lambda) if self.l2_reg_lambda > 0 else None
            self.fc1 = tf.keras.layers.Dense(2048, activation='relu', name='fc1', 
                                           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1), 
                                           bias_initializer=tf.keras.initializers.Constant(0.0),
                                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
            self.dropout_fc1 = tf.keras.layers.Dropout(self.dropout_rate)
            self.fc2 = tf.keras.layers.Dense(2048, activation='relu', name='fc2', 
                                           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1), 
                                           bias_initializer=tf.keras.initializers.Constant(0.0),
                                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
            self.dropout_fc2 = tf.keras.layers.Dropout(self.dropout_rate)
            self.fc3 = tf.keras.layers.Dense(self.num_classes, name='fc3', 
                                           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1), 
                                           bias_initializer=tf.keras.initializers.Constant(0.0),
                                           kernel_regularizer=regularizer, bias_regularizer=regularizer)

        def _convolutional_block_op(self, input_tensor, conv1, bn1, conv2, bn2, pool_layer, training):
            x = conv1(input_tensor)
            x = bn1(x, training=training)
            x = tf.nn.relu(x)
            x = conv2(x)
            x = bn2(x, training=training)
            x = tf.nn.relu(x)
            x = pool_layer(x)
            return x

        def call(self, input_x, training=None):
            embedded_chars = self.embedding_layer(input_x) 
            x = tf.expand_dims(embedded_chars, -1) 
            
            x = self.conv0_layer(x) 
            x = self.bn0_layer(x, training=training)
            x = tf.nn.relu(x)
            
            # Use the layers defined as attributes for convolutional blocks
            x = self._convolutional_block_op(x, self.cb1_conv1, self.cb1_bn1, self.cb1_conv2, self.cb1_bn2, self.cb1_pool, training)
            x = self._convolutional_block_op(x, self.cb2_conv1, self.cb2_bn1, self.cb2_conv2, self.cb2_bn2, self.cb2_pool, training)
            x = self._convolutional_block_op(x, self.cb3_conv1, self.cb3_bn1, self.cb3_conv2, self.cb3_bn2, self.cb3_pool, training)
            x = self._convolutional_block_op(x, self.cb4_conv1, self.cb4_bn1, self.cb4_conv2, self.cb4_bn2, self.cb4_pool, training)
            
            x = tf.transpose(x, [0,3,2,1]) 
            x = tf.squeeze(x, axis=2) 
            x, _ = tf.nn.top_k(x, k=8, name='k-maxpooling') 
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])

            x = self.fc1(x)
            x = self.dropout_fc1(x, training=training)
            x = self.fc2(x) 
            x = self.dropout_fc2(x, training=training)
            logits = self.fc3(x) 
            return logits
    print("--- CharCNN model defined ---")

    # --------------------------------------------------------------------------
    # 2. Data Handling Logic (Test Set)
    # --------------------------------------------------------------------------
    print("\nStep 2: Data Handling - Downloading, saving, and loading DBPedia test data...")
    data_dir = "./dbpedia_data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Ensured directory exists: {data_dir}")

    test_csv_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(test_csv_path):
        print("Downloading DBPedia dataset from Hugging Face (this may take a moment)...")
        ds = load_dataset("DeveloperOats/DBPedia_Classes", trust_remote_code=True)
        print("Dataset downloaded.")
        print(f"Saving 'test' split to {test_csv_path}...")
        with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'title', 'content'])
            for item in ds['test']:
                writer.writerow([str(item['label']), item['title'], item['content']])
        print(f"Test data saved to {test_csv_path}")
        with open(test_csv_path, 'r', newline='', encoding='utf-8') as f_verify:
            reader_verify = csv.reader(f_verify)
            print(f"First 3 lines of {test_csv_path} (including header):")
            for i_verify, row_content_verify in enumerate(reader_verify):
                print(row_content_verify)
                if i_verify >= 2: break
        print("--- Finished saving and verifying test CSV ---")
    else:
        print(f"Test data CSV already exists at {test_csv_path}. Skipping download and save.")

    print(f"Loading data from {test_csv_path}...")
    raw_samples_test = {} 
    with open(test_csv_path, 'r', newline='', encoding='utf-8') as f_read:
        reader = csv.DictReader(f_read)
        for row in reader:
            label = row['class']
            if label not in raw_samples_test:
                raw_samples_test[label] = []
            
            sample_features = np.ones(FEATURE_LEN, dtype=np.int32)
            char_count = 0
            text_fields_to_process = []
            title = row.get('title', "")
            content = row.get('content', "")
            if title: text_fields_to_process.append(title)
            if content: text_fields_to_process.append(content)

            for field_text in text_fields_to_process:
                if char_count >= FEATURE_LEN: break
                for char_val in field_text.lower():
                    if char_val in cdict:
                        if char_count < FEATURE_LEN:
                            sample_features[char_count] = cdict[char_val]
                            char_count += 1
                    if char_count >= FEATURE_LEN: break
            raw_samples_test[label].append(sample_features)

    test_classes_str_list = sorted(list(raw_samples_test.keys()))
    num_classes_for_model = len(test_classes_str_list)
    print(f"Number of classes found in test data: {num_classes_for_model}")

    x_test_list = []
    y_test_indices_list = []
    test_class_name_to_idx_map = {name: i for i, name in enumerate(test_classes_str_list)}

    for class_name_str in test_classes_str_list:
        class_idx = test_class_name_to_idx_map[class_name_str]
        for sample_vec in raw_samples_test[class_name_str]:
            x_test_list.append(sample_vec)
            y_test_indices_list.append(class_idx)
    
    x_test = np.array(x_test_list, dtype=np.int32)
    y_test_indices = np.array(y_test_indices_list, dtype=np.int32)

    if x_test.shape[0] == 0:
        print("Error: Test data (x_test) is empty after processing. Exiting.")
        return
        
    print(f"Shape of x_test: {x_test.shape}, Shape of y_test_indices: {y_test_indices.shape}")
    print("--- Test data loading and initial processing complete ---")
    
    # --------------------------------------------------------------------------
    # 3. Evaluation Logic
    # --------------------------------------------------------------------------
    print("\nStep 3: Evaluation Logic...")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e_gpu: print(e_gpu)
    else:
        print("Running on CPU")

    def batch_iter_test(x_data, batch_size_iter):
        data_size = len(x_data)
        if data_size == 0: 
            print("Warning: batch_iter_test received empty x_data.")
            yield np.array([], dtype=np.int32)
            return
        num_batches_per_epoch = int(data_size / batch_size_iter) + (1 if data_size % batch_size_iter != 0 else 0)
        print(f"Num batches per epoch for test: {num_batches_per_epoch}")
        for batch_idx in range(num_batches_per_epoch):
            start_index = batch_idx * batch_size_iter
            end_index = min((batch_idx + 1) * batch_size_iter, data_size)
            if start_index >= end_index: continue
            yield x_data[start_index:end_index]

    cnn_eval = CharCNN(
        num_classes=num_classes_for_model, 
        dropout_rate=0.0 
    )

    checkpoint_base_dir = './runs_new_tf2_script/' 
    latest_run_dir = None
    checkpoint_to_restore = None

    if os.path.exists(checkpoint_base_dir):
        try:
            all_runs = [d for d in os.listdir(checkpoint_base_dir) if os.path.isdir(os.path.join(checkpoint_base_dir, d))]
            if all_runs:
                timestamp_dirs = [d for d in all_runs if d.isdigit()] 
                if timestamp_dirs:
                    latest_run_timestamp = max(timestamp_dirs)
                    latest_run_dir = os.path.join(checkpoint_base_dir, latest_run_timestamp)
                    checkpoint_dir_eval = os.path.join(latest_run_dir, 'checkpoints')
                    print(f"Attempting to load checkpoints from latest run: {checkpoint_dir_eval}")
                    if os.path.exists(checkpoint_dir_eval):
                        checkpoint_to_restore = tf.train.latest_checkpoint(checkpoint_dir_eval)
                else:
                    print(f"No timestamped run directories found in {checkpoint_base_dir}.")    
            else:
                print(f"No run directories found in {checkpoint_base_dir}.")
        except Exception as e: print(f"Error finding latest checkpoint directory: {e}")
    
    if not checkpoint_to_restore:
        print(f"Error: No checkpoint file found in {checkpoint_base_dir} or its subdirectories. Cannot evaluate.")
        return

    print(f"Using checkpoint file: {checkpoint_to_restore}")

    dummy_input_shape = (1, FEATURE_LEN) 
    dummy_input = np.zeros(dummy_input_shape, dtype=np.int32)
    _ = cnn_eval(tf.constant(dummy_input)) 

    checkpoint_load = tf.train.Checkpoint(model=cnn_eval)
    status = checkpoint_load.restore(checkpoint_to_restore)
    status.expect_partial().assert_existing_objects_matched() 
    print(f"Model restored from {checkpoint_to_restore}.")

    all_predictions = []
    if x_test.shape[0] > 0:
        print("Starting prediction loop...")
        test_batches = batch_iter_test(x_test, batch_size_iter=128)
        for batch_x_data in test_batches:
            if len(batch_x_data) == 0: continue
            x_batch_tf = tf.constant(batch_x_data, dtype=tf.int32)
            logits = cnn_eval(x_batch_tf, training=False)
            batch_predictions = tf.argmax(logits, axis=1).numpy()
            all_predictions.extend(batch_predictions)
        
        all_predictions_np = np.array(all_predictions)

        if len(all_predictions_np) == len(y_test_indices):
            correct_predictions = float(sum(np.equal(all_predictions_np, y_test_indices)))
            accuracy = correct_predictions / float(len(y_test_indices))
            print(f"Total number of test examples: {len(y_test_indices)}")
            print(f"Accuracy: {accuracy:g}")
        else:
            print("Could not calculate accuracy: Number of predictions does not match number of true labels.")
            print(f"Number of predictions: {len(all_predictions_np)}, Number of true labels: {len(y_test_indices)}")
    else:
        print("Skipping evaluation as no test data (x_test) is available.")
    
    print("--- Evaluation complete ---")

if __name__ == '__main__':
    main()
