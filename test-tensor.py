import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# load in the dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

#model = create_model()
# train and save
#model.fit(x_train, y_train, epochs=5)
#model.save("test_model.h5")

# now load the data instead
model = tf.keras.models.load_model("test_model.h5")

# evaluate the results
model.evaluate(x_test, y_test)
