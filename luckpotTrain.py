import os
import matplotlib.pyplot as plt
import tensorflow as tf

# train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
#                                            origin=train_dataset_url)
train_dataset_fp = "./dataset/luckpot_train.csv"
# test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

# test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
#                                   origin=test_url)
test_fp = "./dataset/luckpot_test.csv"
print("Local copy of the dataset file: {}".format(train_dataset_fp))

# CSV文件中列的顺序
column_names = ['num','red_1', 'red_2', 'red_3', 'red_4', 'red_5','red_6','blue_1']

feature_names = column_names[1:]
label_name = column_names[0]

print("FeatureNames: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    
    num_epochs=1)

features, labels = next(iter(train_dataset))

print("Features: {}".format(features))


def pack_features_vector(features, labels):
  """将特征打包到一个数组中"""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

print(features)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(14, activation=tf.nn.relu, input_shape=(7,)),  # 需要给出输入的形式
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(40, activation=tf.nn.relu),
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(30, activation=tf.nn.relu),
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(15, activation=tf.nn.relu),
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


predictions = model(features)
predictions[:7]

tf.nn.softmax(predictions[:7])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels).numpy()))


                                          ## Note: 使用相同的模型变量重新运行此单元

# 保留结果用于绘制
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # 优化模型
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 追踪进度
    epoch_loss_avg(loss_value)  # 添加当前的 batch loss
    # 比较预测标签与真实标签
    epoch_accuracy(y, model(x))

  # 循环结束
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

    
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='num',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# predict_dataset = tf.convert_to_tensor([
#     [5.1, 3.3, 1.7, 0.5,],
#     [5.9, 3.0, 4.2, 1.5,],
#     [6.9, 3.1, 5.4, 2.1]
# ])

# predictions = model(predict_dataset)

# for i, logits in enumerate(predictions):
#   class_idx = tf.argmax(logits).numpy()
#   p = tf.nn.softmax(logits)[class_idx]
#   name = class_names[class_idx]
#   print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))