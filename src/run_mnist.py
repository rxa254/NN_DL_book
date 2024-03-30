import mnist_loader
import network


print("Load the MNIST database...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


print("Initializing the NN...")
net = network.Network([784, 30, 10])

print("Training the NN...")
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print("Done Training and Testing.")