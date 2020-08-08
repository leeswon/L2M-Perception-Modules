import habitat
import numpy as np
from perceptionDataGen import ClassifierDataGenerator

import matplotlib.pyplot as plt
import random
from math import sqrt, floor


object_to_classify = {1: 'wall',
                      4: 'door',
                      17: 'ceiling',
                      2: 'floor',
                      6: 'picture',
                      9: 'window',
                      3: 'chair',
                      8: 'cushion',
                      39: 'objects',
                      28: 'lamp',
                      7: 'cabinet',
                      12: 'curtain',
                      5: 'table',
                      14: 'plant'
}

action_mapping = {
    0: 'stop',
    1: 'move_forward',
    2: 'turn left',
    3: 'turn right'
}

_debug1 = False
_debug2 = False

def semantic_converter(semantic_obs, label_mapping):
    converted_semantic = np.zeros(semantic_obs.shape)
    for row_cnt in range(converted_semantic.shape[0]):
        for col_cnt in range(converted_semantic.shape[1]):
            converted_semantic[row_cnt, col_cnt] = max(label_mapping[semantic_obs[row_cnt, col_cnt]], 0)
    return converted_semantic

def display_observation(rgb_obs, semantic_obs):
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title("RGB")
    plt.imshow(rgb_obs)

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    ax2.set_title("Semantic")
    plt.imshow(semantic_obs)
    plt.show()

def display_sample(train_x, train_y):
    plot_width, plot_height = int(floor(sqrt(train_y.shape[0]))), int(floor(sqrt(train_y.shape[0])))
    obj_class_ids = list(object_to_classify.keys())
    plt.figure()
    for row_cnt in range(plot_height):
        for col_cnt in range(plot_width):
            ax = plt.subplot(plot_height, plot_width, row_cnt*plot_width+col_cnt+1)
            ax.axis('off')
            label_name = object_to_classify[obj_class_ids[train_y[row_cnt*plot_width+col_cnt]]]
            ax.set_title(label_name)
            plt.imshow(train_x[row_cnt*plot_width+col_cnt])
    plt.show()

def example():
    #### Parameters for testing
    env_max_steps = 500
    sampling_period = 4
    num_burnin_ep, num_train_ep = 3, 4
    batch_size = 16
    training_img_size = [64, 64]
    replay_buffer_size = 1000
    #### Parameters for testing

    ################# Place appropriate path to habitat-api here!
    habitat_api_path = '/mnt/Data/Research_Programs/Ubuntu/habitat-api'
    ################# Place appropriate path to habitat-api here!

    config = habitat.get_config(config_paths=habitat_api_path+'/configs/tasks/pointnav_mp3d.yaml')
    config.defrost()
    #config.DATASET.DATA_PATH = habitat_api_path+'/data/datasets/pointnav/mp3d/v1/train/train.json.gz'
    config.DATASET.DATA_PATH = habitat_api_path+'/data/datasets/pointnav/mp3d/v1/val/val.json.gz'
    config.DATASET.SCENES_DIR = habitat_api_path+'/data/scene_datasets/'
    config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'SEMANTIC_SENSOR']
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.freeze()

    env = habitat.Env(config=config)
    scene = env.sim.semantic_annotations()
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    env.episodes = random.sample(env.episodes, num_burnin_ep+num_train_ep)
    print("Environment creation successful")

    data_generator = ClassifierDataGenerator(list(object_to_classify.keys()), img_crop_size=training_img_size, buffer_max_size=replay_buffer_size)
    print("Data generator creation successful")

    ################# Import appropriate learner here!
    learner = None
    if learner is not None:
        print("Lifelong learner creation successful")
        task_info = {"task_index": 0, "task_description": "Matterport3D_classification"}
        learner.addNewTask(task_info=task_info, num_classes=len(object_to_classify))
    ################# Import appropriate learner here!


    for i in range(len(env.episodes)):
        observations = env.reset()
        print("\nEpisode %d"%(i))

        count_steps = 0
        while count_steps < env_max_steps:
            action = random.choices(list(action_mapping.keys()), weights=[1, 33, 33, 33], k=1)[0]
            print("\t", action_mapping[action])
            observations = env.step(action)
            count_steps += 1

            if count_steps % sampling_period == 0:
                data_generator.addSamples(observations['rgb'], semantic_converter(observations['semantic'], instance_id_to_label_id))
                print("\t\t\tGenerator buffer usage : %d/ %d"%(data_generator.getBufferSize(), data_generator.getBufferMaxSize()))
                if _debug1:
                    display_observation(observations['rgb'], observations['semantic'])
                if _debug2 and data_generator.getBufferSize() > batch_size:
                    train_x, train_y = data_generator.getSamples(batch_size)
                    display_sample(train_x, train_y)

            if i >= num_burnin_ep and learner is not None:
                train_x, train_y = data_generator.getSamples(batch_size)
                learner.train(task_info, train_x, train_y)

            if env.episode_over:
                break

    env.close()


if __name__ == "__main__":
    example()