from abc import abstractmethod

class L2MClassifier():
    @abstractmethod
    def __init__(self, model_hyperparam):
        # Initialize hyper-parameters of a lifelong learner
        raise NotImplementedError

    @abstractmethod
    def addNewTask(self, task_info, num_classes):
        # Generate/initialize task-specific sub-modules
        # task_info contains 'task_index' (enumeration of tasks) and 'task_description' (details of task)
        # num_classes is for the output size of task-specific sub-module
        raise NotImplementedError

    @abstractmethod
    def inference(self, task_info, X):
        # Make inference on the given data X according to the task (task_info)
        # return y
        raise NotImplementedError

    @abstractmethod
    def train(self, task_info, X, y):
        # Optimize trainable parameters according to the task (task_info) and data (X and y)
