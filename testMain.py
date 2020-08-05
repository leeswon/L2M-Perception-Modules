import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import random
import numpy as np


action_mapping = {
    0: 'stop',
    1: 'move_forward',
    2: 'turn left',
    3: 'turn right'
}

object_to_classify = [1, 4, 17, 2, 6, 9, 3, 4, 8, 39, 28, 7, 12, 5, 14]
# 1: wall/ 4: door/ 17: ceiling/ 2: floor/ 6: picture/ 9: window/ 3: chair/ 4: door/ 8: cushion/ 39: objects or decorations/ 28: lamp/ 7: cabinet/ 12: curtain/ 5: table/ 14: plant

def semantic_converter(semantic_obs, label_mapping):
    converted_semantic = np.zeros(semantic_obs.shape)
    for row_cnt in range(converted_semantic.shape[0]):
        for col_cnt in range(converted_semantic.shape[1]):
            converted_semantic[row_cnt, col_cnt] = max(label_mapping[semantic_obs[row_cnt, col_cnt]], 0)
    return converted_semantic

def example():
    habitat_api_path = '/mnt/Data/Research_Programs/Ubuntu/habitat-api'
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
    env.episodes = random.sample(env.episodes, 2)

    print("Environment creation successful")








    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = input()

        if keystroke == FORWARD_KEY:
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == LEFT_KEY:
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == RIGHT_KEY:
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == FINISH:
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()