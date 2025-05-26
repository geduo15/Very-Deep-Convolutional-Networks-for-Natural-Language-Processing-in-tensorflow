import csv
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import os
import time
import datetime

def main():
    print("Starting train.py script...")

    # --------------------------------------------------------------------------
    # 1. Data Handling Logic (Constants, Download, Save, Load)
    # --------------------------------------------------------------------------
    print("Step 1: Data Handling - Defining constants, downloading, saving, and loading DBPedia train data...")

    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\\\"/\\\\|_@#$%^&*~`+ =<>()[]{}"
    FEATURE_LEN = 1014 # Max length of a sample
    cdict = {c: i + 2 for i, c in enumerate(ALPHABET)} # Character to index mapping (+2 for padding/unknown)
    print(f"Constants ALPHABET (len: {len(ALPHABET)}), FEATURE_LEN ({FEATURE_LEN}), cdict defined.")

    data_dir = "./dbpedia_data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Ensured directory exists: {data_dir}")

    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv") # Also prepare test path for completeness

    print("Downloading DBPedia dataset from Hugging Face (this may take a moment)...")
    # Using trust_remote_code=True as it might be needed for some datasets.
    ds = load_dataset("DeveloperOats/DBPedia_Classes", trust_remote_code=True)
    print("Dataset downloaded.")

    def save_split_to_csv(dataset_split, csv_path, split_name):
        print(f"Saving '{split_name}' split to {csv_path}...")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'title', 'content'])  # Header row
            for item in dataset_split:
                writer.writerow([str(item['label']), item['title'], item['content']])
        print(f"'{split_name}' data saved to {csv_path}")
        
        # Verify by reading a few lines
        with open(csv_path, 'r', newline='', encoding='utf-8') as f_verify:
            reader_verify = csv.reader(f_verify)
            print(f"First 3 lines of {csv_path} (including header):")
            for i_verify, row_content_verify in enumerate(reader_verify):
                print(row_content_verify)
                if i_verify >= 2:
                    break
        print(f"--- Finished saving and verifying {split_name} CSV ---")

    save_split_to_csv(ds['train'], train_csv_path, "train")
    # Save test data as well, though not strictly used in this training script's main flow
    save_split_to_csv(ds['test'], test_csv_path, "test")


    print(f"Loading data from {train_csv_path} into 'samples' dictionary...")
    samples = {}
    with open(train_csv_path, 'r', newline='', encoding='utf-8') as f_read:
        reader = csv.DictReader(f_read)
        for row in reader:
            label = row['class']
            if label not in samples:
                samples[label] = []
            
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
            samples[label].append(sample_features)

    classes = list(samples.keys())
    class_samples = [samples[c_key] for c_key in classes] # Ensure order matches 'classes'

    print(f"Data loaded into 'samples'. Number of classes: {len(classes)}")
    if classes and samples.get(classes[0]) and samples[classes[0]]:
        print(f"Number of samples for first class ('{classes[0]}'): {len(samples[classes[0]])}")
        if samples[classes[0]][0] is not None:
             print(f"First sample vector for class '{classes[0]}' (first 20 features): {samples[classes[0]][0][:20]}...")
    print("--- Data loading and initial processing complete ---")

    # --------------------------------------------------------------------------
    # 2. Data Preprocessing Logic
    # --------------------------------------------------------------------------
    print("\nStep 2: Data Preprocessing...")

    num_classes_actual = len(classes)
    if num_classes_actual == 0:
        print("Error: No classes found after data loading. Exiting.")
        return

    def build_onehot(input_idx, num_classes_local):
        target = np.zeros(num_classes_local, dtype=np.float32)
        if 0 <= input_idx < num_classes_local:
            target[input_idx] = 1.0
        else:
            print(f"Warning: input_idx {input_idx} is out of bounds for num_classes_local {num_classes_local}")
        return target

    y_one_hot_list = []
    class_name_to_idx_map = {name: i for i, name in enumerate(classes)}

    for class_name in classes:
        class_idx = class_name_to_idx_map[class_name]
        for _ in range(len(samples[class_name])):
            target = build_onehot(class_idx, num_classes_actual)
            y_one_hot_list.append(target)
    
    y = np.array(y_one_hot_list, dtype=np.float32)

    x_flat_list = []
    for class_data in class_samples:
        for sample_item in class_data:
            x_flat_list.append(sample_item)
    x = np.array(x_flat_list, dtype=np.int32)

    if x.shape[0] == 0:
        print("Error: x data is empty after processing. Exiting.")
        return
    if y.shape[0] == 0:
        print("Error: y data is empty after processing. Exiting.")
        return
        
    if x.shape[0] != y.shape[0]:
        print(f"Warning: Mismatch in number of x samples ({x.shape[0]}) and y labels ({y.shape[0]}). Truncating to shorter length.")
        min_len = min(x.shape[0], y.shape[0])
        x = x[:min_len]
        y = y[:min_len]
    
    print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")

    # Reshape y (already done by np.array(y_one_hot_list) if num_classes_actual is correct)
    # y = np.reshape(y, (-1, num_classes_actual)) # This line might be redundant if y is already correct

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    n_dev_samples = min(10000, len(y) // 10)
    if n_dev_samples == 0 and len(y) > 10: # Ensure some dev samples if dataset is reasonably sized
        n_dev_samples = len(y) // 10
    elif n_dev_samples == 0 and len(y) > 0:
        n_dev_samples = 1 # At least one, if possible

    if len(y) > n_dev_samples and n_dev_samples > 0 :
        x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
        y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
    else:
        x_train, y_train = x_shuffled, y_shuffled
        x_dev, y_dev = np.array([]).reshape(0, FEATURE_LEN), np.array([]).reshape(0, num_classes_actual)
    
    print(f"Train/Dev split: {len(y_train)}/{len(y_dev)}")
    print("--- Data preprocessing complete ---")

    # --------------------------------------------------------------------------
    # 3. CharCNN Model Definition
    # --------------------------------------------------------------------------
    print("\nStep 3: Defining CharCNN model...")
    class CharCNN(tf.keras.Model):
        def __init__(self, num_classes=num_classes_actual, sequence_max_length=FEATURE_LEN, 
                     num_quantized_chars=len(ALPHABET) + 2, embedding_size=16, 
                     l2_reg_lambda=0.001, dropout_rate=0.5):
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
            block_configs = [(64, 'cb1'), (128, 'cb2'), (256, 'cb3'), (512, 'cb4')]
            self.conv_blocks = []
            for filters, name_prefix in block_configs:
                block = {
                    'conv1': tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name=name_prefix+'_conv1'),
                    'bn1': tf.keras.layers.BatchNormalization(name=name_prefix+'_bn1'),
                    'conv2': tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name=name_prefix+'_conv2'),
                    'bn2': tf.keras.layers.BatchNormalization(name=name_prefix+'_bn2'),
                    'pool': tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name=name_prefix+'_pool')
                }
                self.conv_blocks.append(block)
            
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

        def _convolutional_block_op(self, input_tensor, block, training):
            x = block['conv1'](input_tensor)
            x = block['bn1'](x, training=training)
            x = tf.nn.relu(x)
            x = block['conv2'](x)
            x = block['bn2'](x, training=training)
            x = tf.nn.relu(x)
            x = block['pool'](x)
            return x

        def call(self, input_x, training=None):
            embedded_chars = self.embedding_layer(input_x) 
            x = tf.expand_dims(embedded_chars, -1) 
            
            x = self.conv0_layer(x) 
            x = self.bn0_layer(x, training=training)
            x = tf.nn.relu(x)
            
            for block_layers in self.conv_blocks:
                x = self._convolutional_block_op(x, block_layers, training)
            
            x = tf.transpose(x, [0,3,2,1]) 
            x = tf.squeeze(x, axis=2) 
            x, _ = tf.nn.top_k(x, k=8, name='k-maxpooling') 
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]]) # Corrected reshape
            
            x = self.fc1(x)
            x = self.dropout_fc1(x, training=training)
            x = self.fc2(x) 
            x = self.dropout_fc2(x, training=training)
            logits = self.fc3(x) 
            return logits

        def loss_fn(self, logits, labels): # Renamed from 'loss' to avoid Keras built-in
            softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            mean_loss = tf.reduce_mean(softmax_loss)
            regularization_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)
            return mean_loss + regularization_loss

        def accuracy_fn(self, logits, labels): # Renamed from 'accuracy'
            predictions = tf.argmax(logits, 1, name="predictions")
            correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    print("--- CharCNN model defined ---")

    # --------------------------------------------------------------------------
    # 4. Training Setup
    # --------------------------------------------------------------------------
    print("\nStep 4: Training Setup...")
    
    # GPU Configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e_gpu:
            print(e_gpu)
    else:
        print("Running on CPU")

    def batch_iter(x_iter, y_iter, batch_size_iter, num_epochs_iter, shuffle=True):
        data_size = len(x_iter)
        if data_size == 0:
            print("Warning: batch_iter received empty data.")
            return
        num_batches_per_epoch = int(data_size / batch_size_iter) + (1 if data_size % batch_size_iter != 0 else 0)
        for epoch_num in range(num_epochs_iter):
            print(f"In epoch >> {epoch_num + 1}")
            print(f"num batches per epoch is: {num_batches_per_epoch}")
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x_shuffled = x_iter[shuffle_indices]
                y_shuffled = y_iter[shuffle_indices]
            else:
                x_shuffled = x_iter
                y_shuffled = y_iter
            for batch_idx in range(num_batches_per_epoch):
                start_index = batch_idx * batch_size_iter
                end_index = min((batch_idx + 1) * batch_size_iter, data_size)
                if start_index >= end_index: continue
                yield x_shuffled[start_index:end_index], y_shuffled[start_index:end_index]

    # Model Instantiation
    # Ensure num_classes_actual is valid before using it
    if num_classes_actual == 0:
        print("Error: num_classes_actual is 0. Cannot instantiate model. Exiting.")
        return
        
    cnn = CharCNN(
        num_classes=num_classes_actual,
        sequence_max_length=FEATURE_LEN,
        num_quantized_chars=len(ALPHABET) + 2,
        embedding_size=16, # Default from notebook
        l2_reg_lambda=0.001, # Default
        dropout_rate=0.5 # Corresponds to keep_prob = 0.5
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_new_tf2_script", timestamp))
    
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.create_file_writer(dev_summary_dir)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn, step=global_step)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    print(f"Checkpoint manager initialized. Checkpoints will be saved in: {checkpoint_dir}")
    print("--- Training setup complete ---")

    # --------------------------------------------------------------------------
    # 5. Train Step and Dev Step Functions
    # --------------------------------------------------------------------------
    print("\nStep 5: Defining train_step and dev_step functions...")
    @tf.function
    def train_step_fn(x_batch_tf, y_batch_tf):
        with tf.GradientTape() as tape:
            logits = cnn(x_batch_tf, training=True)
            loss_value = cnn.loss_fn(logits, y_batch_tf)
        grads = tape.gradient(loss_value, cnn.trainable_variables)
        optimizer.apply_gradients(zip(grads, cnn.trainable_variables))
        global_step.assign_add(1)
        accuracy_value = cnn.accuracy_fn(logits, y_batch_tf)
        return loss_value, accuracy_value

    @tf.function
    def dev_step_fn(x_batch_tf, y_batch_tf):
        logits = cnn(x_batch_tf, training=False)
        loss_value = cnn.loss_fn(logits, y_batch_tf)
        accuracy_value = cnn.accuracy_fn(logits, y_batch_tf)
        return loss_value, accuracy_value
    print("--- train_step and dev_step defined ---")

    # --------------------------------------------------------------------------
    # 6. Main Training Loop
    # --------------------------------------------------------------------------
    print("\nStep 6: Starting Main Training Loop...")
    batch_size_script = 128
    num_epochs_script = 1 # Reduced for quick testing
    evaluate_every_script = 100 # Evaluate more frequently for testing
    checkpoint_every_script = 100

    if len(x_train) == 0 or len(y_train) == 0:
        print("Error: Training data (x_train or y_train) is empty. Skipping training loop.")
        return

    num_batches_per_epoch_script = int(len(x_train) / batch_size_script) + (1 if len(x_train) % batch_size_script != 0 else 0)
    
    train_data_batches = batch_iter(x_train, y_train, batch_size_script, num_epochs_script)

    for x_batch_data, y_batch_data in train_data_batches:
        x_batch_tf = tf.constant(np.array(x_batch_data), dtype=tf.int32)
        y_batch_tf = tf.constant(np.array(y_batch_data), dtype=tf.float32)
        
        loss_train, acc_train = train_step_fn(x_batch_tf, y_batch_tf)
        current_step_val = global_step.numpy()
        
        if current_step_val % (evaluate_every_script // 2) == 0: # Log training more often
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_train, step=current_step_val)
                tf.summary.scalar('accuracy', acc_train, step=current_step_val)
            print(f"{datetime.datetime.now().isoformat()}: step {current_step_val}, train loss {loss_train:.4g}, train acc {acc_train:.4g}")

        if len(x_dev) > 0 and len(y_dev) > 0 and current_step_val % evaluate_every_script == 0:
            print("\nEvaluation:")
            dev_losses, dev_accuracies = [], []
            dev_data_batches = batch_iter(x_dev, y_dev, batch_size_script, 1, shuffle=False)
            for x_dev_batch_data, y_dev_batch_data in dev_data_batches:
                x_dev_tf = tf.constant(np.array(x_dev_batch_data), dtype=tf.int32)
                y_dev_tf = tf.constant(np.array(y_dev_batch_data), dtype=tf.float32)
                loss_dev, acc_dev = dev_step_fn(x_dev_tf, y_dev_tf)
                dev_losses.append(loss_dev.numpy())
                dev_accuracies.append(acc_dev.numpy())
            
            avg_dev_loss = np.mean(dev_losses) if dev_losses else 0
            avg_dev_acc = np.mean(dev_accuracies) if dev_accuracies else 0
            with dev_summary_writer.as_default():
                tf.summary.scalar('loss', avg_dev_loss, step=current_step_val)
                tf.summary.scalar('accuracy', avg_dev_acc, step=current_step_val)
            print(f"Dev eval: step {current_step_val}, dev loss {avg_dev_loss:.4g}, dev acc {avg_dev_acc:.4g}")
            print(f"Epoch: {int(current_step_val / num_batches_per_epoch_script)}\n")

        if current_step_val % checkpoint_every_script == 0:
            save_path = manager.save()
            print(f"Saved model checkpoint to {save_path} for step {current_step_val}\n")
            
    print("--- Training loop complete ---")

if __name__ == '__main__':
    main()
