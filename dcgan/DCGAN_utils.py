from dcgan.dcgan import discriminator_model
from dcgan.dcgan_cifar10 import discriminator_model_cifar10


class DCGAN:
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.model = discriminator_model()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('dcgan/discriminator_mnist')
        elif dataset == 'CIFAR10':
            self.model = discriminator_model_cifar10()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('dcgan/discriminator_cifar10')
        elif dataset == 'FM':
            self.model = discriminator_model()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('dcgan/discriminator_fm')
        elif dataset == 'SVHN':
            self.model = discriminator_model_cifar10()
            self.model.compile(loss='binary_crossentropy', optimizer="SGD")
            self.model.load_weights('dcgan/discriminator_svhn')

    def predict_batch(self, preprocessed_test_inputs):
        result = self.model.predict(preprocessed_test_inputs)
        return result
