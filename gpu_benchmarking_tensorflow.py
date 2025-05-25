import tensorflow as tf
import time

# Check device
print(tf.config.list_physical_devices('GPU'))

# Synthetic data
batch_size = 64
input_shape = (batch_size, 224, 224, 3)
x = tf.random.normal(input_shape)
y = tf.random.uniform((batch_size,), maxval=1000, dtype=tf.int32)

model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=1000)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

# Warm-up
model.train_on_batch(x, y)

# Benchmark
start = time.time()
for _ in range(20):
    model.train_on_batch(x, y)
end = time.time()

print(f"Time taken for 20 iterations: {end - start:.2f} seconds")
