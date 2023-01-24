from keras.datasets import cifar10
import numpy as np
from PIL import Image
import argparse
import math
import keras
from keras import layers, models
import numpy as np

latent_dim = 32
channels = 3
height = 32
width = 32


def generator_model():
    generator_input = keras.Input(shape=(latent_dim,))

    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)

    x = layers.Conv2D(256, 5, padding='same')(x)  # 16*16*256
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D()(x)
    # x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x) # 32*32*256 upsampling
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)  # sample 32*32*3 image

    return models.Model(generator_input, x)


def discriminator_model_cifar10():
    descriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(128, 3)(descriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(descriminator_input, x)


def generator_containing_discriminator(g, d):
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = d(g(gan_input))
    gan = models.Model(gan_input, gan_output)
    return gan


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape((X_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
    X_test = X_test[:, :, :, None]
    d = discriminator_model_cifar10()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    d_optim = keras.optimizers.Adam(lr=0.0002, clipvalue=1.0, decay=1e-8)
    g_optim = keras.optimizers.Adam(lr=0.0002, clipvalue=1.0, decay=1e-8)

    g.compile(loss='binary_crossentropy', optimizer="Adam")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(optimizer=d_optim, loss='binary_crossentropy')

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 32))
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 32))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
        if (epoch + 1) % 4 == 0:
            g.save_weights('models_cifar10/generator_epoch' + str(epoch + 1), True)
            d.save_weights('models_cifar10/discriminator_epoch' + str(epoch + 1), True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model_cifar10()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE * 20, 100))
        generated_images = g.predict(noise, verbose=1)

        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        print(pre_with_index)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        print(generated_images.shape)
        image = combine_images(generated_images)
    image = image * 255
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def discriminate(imgs):
    d = discriminator_model_cifar10()
    d.compile(loss='binary_crossentropy', optimizer="SGD")
    d.load_weights('discriminator')

    return d.predict(imgs, verbose=1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
