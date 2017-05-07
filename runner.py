import optparse
import network2 as nt
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
    parser.add_option('-o', dest="outFile", type="string", \
                      help="Include to dictate the filename to write with training iteration information", \
                      default=None)
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
            print("Need to specify InitializingType with a -t <type> option.")
            exit(0)
        # Exit without doing anything further
        exit(0)

    # load the mnist data from /data/mnist.pkl.gz
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # if the user specified a weight initalization file setup the network with those weights
    # otherwise use the appropriate weight initializer
    if not (options.initializerLoad == None):
        if (options.initializingType == "LargeWeightInitializer"):
            net = nt.Network([784, 100, 10], cost=nt.QuadraticCost, initializer=nt.LargeWeightInitializer)
        else:
            net = nt.Network([784, 100, 10], cost=nt.QuadraticCost, initializer=nt.DefaultWeightInitializer)
        net.load(options.initializerLoad)
    elif (options.initializingType == "LargeWeightInitializer"):
        net = nt.Network([784, 100, 10], cost=nt.QuadraticCost, initializer=nt.LargeWeightInitializer)
    else:
        net = nt.Network([784, 100, 10], cost=nt.QuadraticCost, initializer=nt.DefaultWeightInitializer)

    print ("Options Selected:")
    if (options.initializerLoad):
        print("Network Used: {}".format(options.initializerLoad))
    print("Epochs: {:d}".format(options.epochs))
    print("Regularizer: {}".format(options.regularizer))


    if (options.regularizer == "MaxFixed"):
        net.SGD(training_data, options.epochs, 10, 0.5, lmbda=10, evaluation_data=test_data,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                output_file_name=options.outFile)
    elif (options.regularizer == "MinFixed"):
        net.SGD(training_data, options.epochs, 10, 0.5, lmbda=0.1, evaluation_data=test_data,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                output_file_name=options.outFile)
    elif (options.regularizer == "Linear"):
        for i in range(1, options.epochs):
            if i % 10 == 0:
                counter += 1;
                net.SGD(training_data, 1, 10, 0.5, lmbda=10/counter, evaluation_data=test_data,
                        monitor_evaluation_accuracy=True,
                        monitor_evaluation_cost=True,
                        monitor_training_accuracy=True,
                        monitor_training_cost=True,
                        output_file_name=options.outFile)
    elif (options.regularizer == "Exponential"):
        for i in range(1, options.epochs):
            if i % 10 == 0:
                counter += 1;
                net.SGD(training_data, 1, 10, 0.5, lmbda=10/(counter^2), evaluation_data=test_data,
                        monitor_evaluation_accuracy=True,
                        monitor_evaluation_cost=True,
                        monitor_training_accuracy=True,
                        monitor_training_cost=True,
                        output_file_name=options.outFile)
    else:
        print("Please specify a correct regularizer option for -r")
        exit(0)

if __name__ == "__main__":
    Main()




