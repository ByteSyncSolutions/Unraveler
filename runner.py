import optparse
from network2 import Network as nt
import mnist_loader

# Command Line Interface Options and Runner

# Need to add a way to capture the result data and place in a file for analysis.

def Main():

    counter = 0

    parser = optparse.OptionParser()
    parser.add_option('-i', dest="initializing", type="string", \
                      help="Include (Y) to dictate that network just initializes and saves network model")
    parser.add_option('-t', dest="initializingType", type="string", \
                      help="Include to dictate which Intializer function to use (LargeWeightInitializer) | (DefaultWeightInitializer)")
    parser.add_option('-f', dest="initializerLoad", type="string", \
                      help="Include to dictate which Network to use and run Network")
    parser.add_option('-e', dest="epochs", type="int", \
                      help="Include to specify the number of epochs to have the network run through")
    parser.add_option('-r', dest="regularizer", type="string", \
                      help="Include to dictate what type of regularization to perform on the Network. Types include \
                           (MaxFixed) | (MinFIxed) | (Linear) | (Exponential)")

    (options, args) = parser.parse_args()

    if (options.initializing == "Y"):
        if (options.initializingType == "LargeWeightInitializer"):
            net = nt.Network([784, 100, 10], cost=nt.QuadraticCost, initializer=nt.LargeWeightInitializer)
            filename = input("Please indicate filename to save Network as:")
            net.save(filename)
        elif (options.initializingType == "DefaultWeightInitializer"):
            net = nt.Network([784, 100, 10], cost=nt.QuadraticCost, initializer=nt.DefaultWeightInitializer)
            filename = input("Please indicate filename to save Network as:")
            net.save(filename)
        else:
            print("Need to specify InitializingType with a -f <type> option.")
            exit(0)
        # Exit without doing anything further
        exit(0)

    if not (options.initializerLoad == None):

        print ("Options Selected:")
        print("Network Used: {}").format(options.initializerLoad)
        print("Epochs: {}").format(options.epochs)
        print("Regularizer: {}").format(options.regularizer)

        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        net = nt.load(options.initializerLoad)

        if (options.regularizer == "MaxFixed"):
            net.SGD(training_data, 400, 10, 0.5, lmda=10, evaluation_data=test_data, monitor_evaluation_accuracy=True,
                    monitor_training_accuracy=True)
        elif (options.regularizer == "MinFixed"):
            net.SGD(training_data, 400, 10, 0.5, lmda=0.1, evaluation_data=test_data, monitor_evaluation_accuracy=True,
                    monitor_training_accuracy=True)
        elif (options.regularizer == "Linear"):
            for i in range(0,399):
                if i % 10 == 0:
                    counter += 1;
                    net.SGD(training_data, 1, 10, 0.5, lmda=10/i, evaluation_data=test_data, monitor_evaluation_accuracy=True,
                        monitor_training_accuracy=True)
        elif (options.regularizer == "Exponential"):
            for i in range(0,399):
                if i % 10 == 0:
                    counter += 1;
                    net.SGD(training_data, 1, 10, 0.5, lmda=10/(i^2), evaluation_data=test_data, monitor_evaluation_accuracy=True,
                    monitor_training_accuracy=True)
        else:
            print("Please specify a correct regularizer option for -r")
            exit(0)

if __name__ == "__main__":
    Main()




