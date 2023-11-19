import tensorflow as tf
import larq as lq


# Data Pre-Processing
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape the images to the format (28, 28, 1) as the MNIST images are grayscale
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Defning the model structure
# Following Larq's documentation model depth and parameters
model = tf.keras.models.Sequential()

# The first layer, only the weights are quantized while activations are left full-precision
model.add(lq.layers.QuantConv2D(32, (3, 3),
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=(28, 28, 1)))

model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

# The second layer has both quantized weights and activations using the Straight-through-estimator sign activation technquie.
# Using straight-through-estimator to overcome undifferentiability issues
model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, input_quantizer="ste_sign",kernel_quantizer="ste_sign",kernel_constraint="weight_clip"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

# The third layer following the second layer
model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, input_quantizer="ste_sign",kernel_quantizer="ste_sign",kernel_constraint="weight_clip"))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Flatten())
# The fourth layer
model.add(lq.layers.QuantDense(64, use_bias=False, input_quantizer="ste_sign",kernel_quantizer="ste_sign",kernel_constraint="weight_clip"))
model.add(tf.keras.layers.BatchNormalization(scale=False))
# The fifth layer
model.add(lq.layers.QuantDense(10, use_bias=False, input_quantizer="ste_sign",kernel_quantizer="ste_sign",kernel_constraint="weight_clip"))
model.add(tf.keras.layers.BatchNormalization(scale=False))
#Output layer, multi class classification
model.add(tf.keras.layers.Activation("softmax"))



#Compile and train the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=6)

# Evaluate the network
lq.models.summary(model)
test_loss, test_acc = model.evaluate(test_images, test_labels)
