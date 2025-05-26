#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset
import os
import csv

print("Downloading and preparing DBPedia dataset...")
ds = load_dataset("DeveloperOats/DBPedia_Classes")

data_dir = "/dbpedia_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_csv_path = os.path.join(data_dir, "train.csv")

with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for item in ds['train']:
        # Format: class_label, title, content
        # The original code expects class label as string, and then fields for title and content
        writer.writerow([str(item['label']), item['title'], item['content']])
print(f"Train data saved to {train_csv_path}")

print("Dataset preparation complete.")


# In[ ]:


import csv
import numpy as np
samples = {}
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
FEATURE_LEN =1014
cdict = {}
for i,c in enumerate(ALPHABET):
    cdict[c] = i + 2


# In[ ]:


samples = {}
with open('/dbpedia_data/train.csv') as f:
    reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
    for row in reader:
        label = row['class']
        if label not in samples:
            samples[label] = []
        sample = np.ones(FEATURE_LEN)
        count = 0
        # Ensure row['fields'] is not None before iterating
        if row['fields'] is not None:
            for field in row['fields']:
                if field is not None: # Ensure field itself is not None
                    for char in field.lower():
                        if char in cdict:
                            sample[count] = cdict[char]
                            count += 1
                        if count >= FEATURE_LEN-1:
                            break
                if count >= FEATURE_LEN-1:
                    break
        samples[label].append(sample)
    samples_per_class = None
    classes = list(samples.keys()) # Convert to list for consistent ordering if needed
    class_samples = []
    for c in classes:
        if samples_per_class is None:
            # samples_per_class = len(samples[c]) # This was for balanced classes
            pass # For DBPedia from HF, classes might not be perfectly balanced initially
        # else:
            # assert samples_per_class == len(samples[c]) # Assertion removed
        class_samples.append(samples[c])


# In[ ]:


def build_onehot(input_idx, num_classes_local):
    target = np.zeros(num_classes_local)
    target[input_idx] = 1
    return target

num_classes_actual = len(classes)
y= []
class_name_to_idx_map_train = {name: i for i, name in enumerate(classes)}

for class_name in classes:
    class_idx = class_name_to_idx_map_train[class_name]
    for _ in range(len(samples[class_name])):
        target = build_onehot(class_idx, num_classes_actual)
        y.append(target)
y = np.array(y)


# In[ ]:


# Flatten class_samples before reshaping if it's a list of lists of varying lengths
x_flat_list = []
for class_data in class_samples:
    for sample_item in class_data:
        x_flat_list.append(sample_item)
x = np.array(x_flat_list)
if x.shape[0] != len(y):
    # This might happen if sample collection logic has issues or data is very skewed
    print(f"Warning: Mismatch in number of x samples ({x.shape[0]}) and y labels ({len(y)})")
    # Attempt to truncate the longer one, though this indicates a deeper issue
    min_len = min(x.shape[0], len(y))
    x = x[:min_len]
    y = y[:min_len]
    print(f"Adjusted to {min_len} samples.")


# In[ ]:


y = np.reshape(y,(-1,num_classes_actual))


# In[ ]:


np.random.seed(10)
if len(x) == 0:
    print("Error: No data loaded into x. Cannot proceed with shuffling and splitting.")
    # Handle error appropriately, e.g., by exiting or raising an exception
    # For now, create dummy data to allow script to run further for structure testing
    # x_train, x_dev, y_train, y_dev = np.array([]), np.array([]), np.array([]), np.array([])
else:
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # Ensure n_dev_samples is not larger than available samples
    n_dev_samples = min(10000, len(y) // 10) # Use 10% or 10000, whichever is smaller
    if n_dev_samples == 0 and len(y) > 0: # Ensure at least 1 dev sample if data exists
        n_dev_samples = 1
    
    if len(y) > n_dev_samples:
        x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
        y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
    else: # Not enough data for a dev split, use all for training
        x_train, y_train = x_shuffled, y_shuffled
        x_dev, y_dev = np.array([]).reshape(0, FEATURE_LEN if FEATURE_LEN else 1014), np.array([]).reshape(0, num_classes_actual if num_classes_actual else 14) # Empty arrays with correct shape
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# In[ ]:


def Convolutional_Block(input_tensor, filter_num, train, scope):
    norm_init = tf.keras.initializers.RandomNormal(stddev=0.05)
    
    # Define layers within the scope for clarity, though Keras handles naming
    conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,1), strides=(1,1), padding="SAME", kernel_initializer=norm_init, name=scope+"_conv1")(input_tensor)
    bn1 = tf.keras.layers.BatchNormalization(name=scope+"_BN1")(conv1, training=train)
    relu1 = tf.keras.layers.ReLU(name=scope+"_relu1")(bn1)
    
    conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,1), strides=(1,1), padding="SAME", kernel_initializer=norm_init, name=scope+"_conv2")(relu1)
    bn2 = tf.keras.layers.BatchNormalization(name=scope+"_BN2")(conv2, training=train)
    relu2 = tf.keras.layers.ReLU(name=scope+"_relu2")(bn2)
    
    pooled = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name=scope+"_pool1")(relu2)
    return pooled

def Conv(input_tensor, filter_num, kernel_h_size, train, scope): # Assuming kernel width is always 1 based on context, and strides on embedding dim is 1
    norm_init = tf.keras.initializers.RandomNormal(stddev=0.05)
    conv = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(kernel_h_size, input_tensor.shape[2]), strides=(1,1), padding="SAME", kernel_initializer=norm_init, name=scope+"_conv")(input_tensor)
    bn = tf.keras.layers.BatchNormalization(name=scope+"_BN")(conv, training=train)
    return bn

def linear(input_tensor, output_dim, scope=None, stddev=0.1, l2_lambda=0.0): # Added l2_lambda for regularization
    norm_init = tf.keras.initializers.RandomNormal(stddev=stddev)
    const_init = tf.keras.initializers.Constant(0.0)
    regularizer = tf.keras.regularizers.L2(l2_lambda) if l2_lambda > 0 else None
    
    dense_layer = tf.keras.layers.Dense(units=output_dim, 
                                     kernel_initializer=norm_init, 
                                     bias_initializer=const_init, 
                                     kernel_regularizer=regularizer,
                                     bias_regularizer=regularizer, # Also apply to bias as in original l2_loss = tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
                                     name=scope)
    output = dense_layer(input_tensor)
    return output # Return only the output tensor


# In[ ]:


import tensorflow as tf
class CharCNN(tf.keras.Model):
    """
    A CNN for text classification, implemented as a tf.keras.Model.
    Based on the Very Deep Convolutional Networks for Natural Language Processing.
    """
    def __init__(self, num_classes=14, sequence_max_length=1014, num_quantized_chars=71, 
                 embedding_size=16, l2_reg_lambda=0.001, dropout_rate=0.0):
        super(CharCNN, self).__init__()
        self.num_classes = num_classes
        self.sequence_max_length = sequence_max_length
        self.num_quantized_chars = num_quantized_chars
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout_rate = dropout_rate 

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.num_quantized_chars, 
                                                         output_dim=self.embedding_size, 
                                                         input_length=self.sequence_max_length,
                                                         embeddings_initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
                                                         name='embedding_W')
        self.conv0_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, self.embedding_size), strides=(1,1), padding="SAME", 
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='conv0')
        self.bn0_layer = tf.keras.layers.BatchNormalization(name='bn0')

        self.conv_block1_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb1_conv1')
        self.conv_block1_bn1 = tf.keras.layers.BatchNormalization(name='cb1_bn1')
        self.conv_block1_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb1_conv2')
        self.conv_block1_bn2 = tf.keras.layers.BatchNormalization(name='cb1_bn2')
        self.conv_block1_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb1_pool')

        self.conv_block2_conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb2_conv1')
        self.conv_block2_bn1 = tf.keras.layers.BatchNormalization(name='cb2_bn1')
        self.conv_block2_conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb2_conv2')
        self.conv_block2_bn2 = tf.keras.layers.BatchNormalization(name='cb2_bn2')
        self.conv_block2_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb2_pool')

        self.conv_block3_conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb3_conv1')
        self.conv_block3_bn1 = tf.keras.layers.BatchNormalization(name='cb3_bn1')
        self.conv_block3_conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb3_conv2')
        self.conv_block3_bn2 = tf.keras.layers.BatchNormalization(name='cb3_bn2')
        self.conv_block3_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb3_pool')

        self.conv_block4_conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb4_conv1')
        self.conv_block4_bn1 = tf.keras.layers.BatchNormalization(name='cb4_bn1')
        self.conv_block4_conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,1), padding='SAME', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05), name='cb4_conv2')
        self.conv_block4_bn2 = tf.keras.layers.BatchNormalization(name='cb4_bn2')
        self.conv_block4_pool = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding='SAME', name='cb4_pool')
        
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
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) 
        h0 = self.conv0_layer(embedded_chars_expanded) 
        h0 = self.bn0_layer(h0, training=training)
        h0 = tf.nn.relu(h0)
        h1 = self._convolutional_block_op(h0, self.conv_block1_conv1, self.conv_block1_bn1, self.conv_block1_conv2, self.conv_block1_bn2, self.conv_block1_pool, training)
        h2 = self._convolutional_block_op(h1, self.conv_block2_conv1, self.conv_block2_bn1, self.conv_block2_conv2, self.conv_block2_bn2, self.conv_block2_pool, training)
        h3 = self._convolutional_block_op(h2, self.conv_block3_conv1, self.conv_block3_bn1, self.conv_block3_conv2, self.conv_block3_bn2, self.conv_block3_pool, training)
        h4 = self._convolutional_block_op(h3, self.conv_block4_conv1, self.conv_block4_bn1, self.conv_block4_conv2, self.conv_block4_bn2, self.conv_block4_pool, training)
        h5_transposed = tf.transpose(h4, [0,3,2,1]) 
        h5_squeezed = tf.squeeze(h5_transposed, axis=2) 
        pooled_values, _ = tf.nn.top_k(h5_squeezed, k=8, name='k-maxpooling') 
        h6_flattened = tf.reshape(pooled_values, [-1, 512 * 8])
        fc1_out = self.fc1(h6_flattened)
        fc1_out = self.dropout_fc1(fc1_out, training=training)
        fc2_out = self.fc2(fc1_out) 
        fc2_out = self.dropout_fc2(fc2_out, training=training)
        logits = self.fc3(fc2_out) 
        return logits

    def loss(self, logits, labels):
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mean_loss = tf.reduce_mean(softmax_loss)
        regularization_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)
        return mean_loss + regularization_loss 

    def accuracy(self, logits, labels):
        predictions = tf.argmax(logits, 1, name="predictions")
        correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



# In[ ]:


# Ensure x_train and y_train are not empty before trying to access batch from them
if len(x_train) > 0 and len(y_train) > 0:
    x_batch = x_train[0:min(64, len(x_train))]
    y_batch = y_train[0:min(64, len(y_train))]
else:
    # Handle empty training data case - create dummy batch for structure test if needed
    print("Warning: Training data is empty. Using dummy batch for initial test run.")
    # These dummy values should align with expected shapes and types
    dummy_feature_len = FEATURE_LEN if 'FEATURE_LEN' in globals() else 1014
    dummy_num_classes = num_classes_actual if 'num_classes_actual' in globals() and num_classes_actual > 0 else 14
    x_batch = np.zeros((1, dummy_feature_len), dtype=np.int32)
    y_batch = np.zeros((1, dummy_num_classes), dtype=np.float32)
    if dummy_num_classes > 0:
        y_batch[0,0] = 1.0 # Ensure valid one-hot for dummy


# In[ ]:


import os
import tensorflow as tf
import numpy as np
import datetime # Make sure datetime is imported for the training loop cell

# Configure GPU (example, adjust as needed)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0') 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Running on GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("Running on CPU")

# Instantiate the model
num_classes_runtime = y_batch.shape[1]
sequence_max_length_runtime = x_batch.shape[1]
num_quantized_chars_runtime = len(ALPHABET) + 2 
embedding_size_val = 16 
l2_lambda_val = 0.001
dropout_rate_val = 0.5 # This was dropout_keep_prob=0.5, so rate is 0.5

cnn_instance = CharCNN(num_classes=num_classes_runtime, 
              sequence_max_length=sequence_max_length_runtime, 
              num_quantized_chars=num_quantized_chars_runtime, 
              embedding_size=embedding_size_val,
              l2_reg_lambda=l2_lambda_val,
              dropout_rate=dropout_rate_val)

x_batch_tf = tf.constant(x_batch, dtype=tf.int32)
y_batch_tf = tf.constant(y_batch, dtype=tf.float32)

# Build the model by calling it once (important for subclassed Keras models)
_ = cnn_instance(x_batch_tf, training=True) # training=True for initial build with dropout/BN active

logits = cnn_instance(x_batch_tf, training=True) # Call with training=True for this test run
acc_value = cnn_instance.accuracy(logits, y_batch_tf)
loss_value = cnn_instance.loss(logits, y_batch_tf)

print(f"Initial Test Run -> Accuracy: {acc_value.numpy()}")
print(f"Initial Test Run -> Loss: {loss_value.numpy()}")


# In[ ]:


import os
import time
import datetime
def batch_iter(x_iter, y_iter, batch_size_iter, num_epochs_iter, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(x_iter)
    if data_size == 0: # Handle empty data case
      print("Warning: batch_iter received empty data.")
      return
    num_batches_per_epoch = int(data_size/batch_size_iter) + (1 if data_size % batch_size_iter != 0 else 0)
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
            if start_index >= end_index: # Avoid empty batches
                continue
            x_batch_out = x_shuffled[start_index:end_index]
            y_batch_out = y_shuffled[start_index:end_index]
            yield list(zip(x_batch_out, y_batch_out))


# In[ ]:


batch_size_train = 128
num_epochs_train = 1 # Reduced for quick testing
evaluate_every_steps = 300
checkpoint_every_steps = 300
dropout_rate_model_train = 0.5 # Original dropout_keep_prob was 0.5

# Ensure y_train is not empty for num_classes_model
if len(y_train) == 0:
    print("Error: y_train is empty. Cannot determine number of classes for the model.")
    # num_classes_model = 14 # Fallback, but indicates an issue
else:
    num_classes_model = y_train.shape[1]

if len(x_train) > 0:
    sequence_max_length_model = x_train.shape[1]
else:
    print("Error: x_train is empty. Cannot determine sequence_max_length for the model.")
    # sequence_max_length_model = FEATURE_LEN # Fallback

if 'num_classes_model' in locals() and 'sequence_max_length_model' in locals():
    cnn_train_model = CharCNN(num_classes=num_classes_model, 
                  sequence_max_length=sequence_max_length_model, 
                  num_quantized_chars=len(ALPHABET) + 2, 
                  embedding_size=16,
                  l2_reg_lambda=0.001,
                  dropout_rate=dropout_rate_model_train)

    global_step_var = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=1e-3)

    ts = str(int(time.time()))
    output_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_new_tf2", ts))
    print(f"Writing to {output_dir}\n")

    train_log_dir = os.path.join(output_dir, "summaries", "train")
    train_summary_writer_inst = tf.summary.create_file_writer(train_log_dir)
    dev_log_dir = os.path.join(output_dir, "summaries", "dev")
    dev_summary_writer_inst = tf.summary.create_file_writer(dev_log_dir)

    ckpt_dir = os.path.abspath(os.path.join(output_dir, "checkpoints"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.Checkpoint(optimizer=optimizer_adam, model=cnn_train_model, step=global_step_var)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)

    @tf.function
    def train_step_fn(x_tensor, y_tensor):
        with tf.GradientTape() as tape_grads:
            logits_pred = cnn_train_model(x_tensor, training=True)
            current_loss = cnn_train_model.loss(logits_pred, y_tensor)
        gradients = tape_grads.gradient(current_loss, cnn_train_model.trainable_variables)
        optimizer_adam.apply_gradients(zip(gradients, cnn_train_model.trainable_variables), global_step=global_step_var)
        current_accuracy = cnn_train_model.accuracy(logits_pred, y_tensor)
        return current_loss, current_accuracy

    @tf.function
    def dev_step_fn(x_tensor, y_tensor):
        logits_pred = cnn_train_model(x_tensor, training=False)
        current_loss = cnn_train_model.loss(logits_pred, y_tensor)
        current_accuracy = cnn_train_model.accuracy(logits_pred, y_tensor)
        return current_loss, current_accuracy

    if len(x_train) > 0 :
      num_batches_epoch = int(len(x_train)/batch_size_train) + (1 if len(x_train) % batch_size_train != 0 else 0)
      train_batches = batch_iter(x_train, y_train, batch_size_train, num_epochs_train)
      for train_batch_data in train_batches:
          x_train_batch, y_train_batch = zip(*train_batch_data)
          x_tf = tf.constant(np.array(x_train_batch), dtype=tf.int32)
          y_tf = tf.constant(np.array(y_train_batch), dtype=tf.float32)
          loss_tr, acc_tr = train_step_fn(x_tf, y_tf)
          step_val = global_step_var.numpy()
          if step_val % (evaluate_every_steps // 2) == 0:
              with train_summary_writer_inst.as_default():
                  tf.summary.scalar('loss', loss_tr, step=step_val)
                  tf.summary.scalar('accuracy', acc_tr, step=step_val)
              print(f"{datetime.datetime.now().isoformat()}: step {step_val}, train loss {loss_tr:.4g}, train acc {acc_tr:.4g}")
          if len(x_dev) > 0 and step_val % evaluate_every_steps == 0:
              print("\nEvaluation:")
              dev_losses_list, dev_accs_list = [], []
              dev_batches = batch_iter(x_dev, y_dev, batch_size_train, 1, shuffle=False)
              for dev_batch_data in dev_batches:
                  x_dev_b, y_dev_b = zip(*dev_batch_data)
                  x_dev_tf_b = tf.constant(np.array(x_dev_b), dtype=tf.int32)
                  y_dev_tf_b = tf.constant(np.array(y_dev_b), dtype=tf.float32)
                  loss_dv, acc_dv = dev_step_fn(x_dev_tf_b, y_dev_tf_b)
                  dev_losses_list.append(loss_dv.numpy())
                  dev_accs_list.append(acc_dv.numpy())
              with dev_summary_writer_inst.as_default():
                  tf.summary.scalar('loss', np.mean(dev_losses_list), step=step_val)
                  tf.summary.scalar('accuracy', np.mean(dev_accs_list), step=step_val)
              print(f"Dev eval: step {step_val}, dev loss {np.mean(dev_losses_list):.4g}, dev acc {np.mean(dev_accs_list):.4g}")
              print(f"Epoch: {int(step_val / num_batches_epoch)}\n")
          if step_val % checkpoint_every_steps == 0:
              ckpt_save_path = ckpt_manager.save()
              print(f"Saved model checkpoint to {ckpt_save_path} for step {step_val}\n")
    else:
        print("Skipping training loop as x_train is empty.")
else:
    print("Skipping model training due to missing num_classes_model or sequence_max_length_model.")


# In[ ]:




