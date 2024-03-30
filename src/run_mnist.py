import yaml
import mnist_loader
import network
from timeit import default_timer as timer

# Load configuration from the YAML file
with open('network_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("Loaded the MNIST database...")



# Initialize the Network with configurations from YAML file
net = network.Network(config['network_structure'], config['activation_functions'])
print("Thinking Machine Awakened...")


print("Training the NN...")
t0 = timer()
# Using parameters from the configuration file
net.SGD(training_data, config['epochs'], config['mini_batch_size'], config['learning_rate'], test_data=test_data)

print("Done Training and Testing Ex Machina.")
print("Trainging time = {:0.1f}".format(timer() - t0) + " seconds.")