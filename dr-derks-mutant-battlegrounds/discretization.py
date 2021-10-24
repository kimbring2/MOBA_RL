import random
import numpy as np
import gym


class SmartDiscrete:
    def __init__(self, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = []

        # 0=donâ€™t cast. 1-3=cast corresponding ability
        CastingSlot_list = [1,2,3] 

        # 0=keep current focus. 1=focus home statue. 2-3=focus teammates, 4=focus enemy statue, 5-7=focus enemy
        ChangeFocus_list = [1,2,3,4,5,6,7] 
        
        self.all_actions_dict = {}
        list_num = 5

        self.all_actions_dict[(0.7, 0.0, 0.0, 0, 0)] = 0
        self.all_actions_dict[(-0.7, 0.0, 0.0, 0, 0)] = 1
        self.all_actions_dict[(0.0, 0.3, 0.0, 0, 0)] = 2
        self.all_actions_dict[(0.0, -0.3, 0.0, 0, 0)] = 3
        self.all_actions_dict[(0.0, 0.0, 0.7, 0, 0)] = 4

        for CastingSlot_value in CastingSlot_list:
            self.all_actions_dict[(0.0, 0.0, 0.0, CastingSlot_value, 0)] = list_num
            list_num += 1

        for ChangeFocus_value in ChangeFocus_list:
            #print("ChangeFocus_value: " + str(ChangeFocus_value))
            self.all_actions_dict[(0.0, 0.0, 0.0, 0, ChangeFocus_value)] = list_num
            list_num += 1

        self.ignore_keys = ignore_keys
        self.key_to_dict = {
            0: (0.7, 0.0, 0.0, 0, 0),
            1: (-0.7, 0.0, 0.0, 0, 0),
            2: (0.0, 0.3, 0.0, 0, 0),
            3: (0.0, -0.3, 0.0, 0, 0),
            4: (0.0, 0.0, 0.7, 0, 0),

            5: (0.0, 0.0, 0.0, 1, 0),
            6: (0.0, 0.0, 0.0, 2, 0),
            7: (0.0, 0.0, 0.0, 3, 0),

            8: (0.0, 0.0, 0.0, 0, 1),
            9: (0.0, 0.0, 0.0, 0, 2),
            10: (0.0, 0.0, 0.0, 0, 3),
            11: (0.0, 0.0, 0.0, 0, 4),
            12: (0.0, 0.0, 0.0, 0, 5),
            13: (0.0, 0.0, 0.0, 0, 6),
            14: (0.0, 0.0, 0.0, 0, 7),
        }

    @staticmethod
    def discrete_camera(camera):
        result = list(camera)
        if abs(result[1]) >= abs(result[0]):
            result[0] = 0
        else:
            result[1] = 0

        def cut(value, max_value=1.2):
            sign = -1 if value < 0 else 1
            if abs(value) >= max_value:
                return 5 * sign
            else:
                return 0

        cutten = list(map(cut, result))
        return cutten

    def preprocess_action_dict(self, action_dict):
        #print("action_dict: " + str(action_dict))

        if random.random() <= 0.5:
            MoveX = random.uniform(-1.0, 1.0)
            Rotate = random.uniform(-1.0, 1.0)
            ChaseFocus = random.uniform(0.0, 1.0)
            CastingSlot = 0
            ChangeFocus = 0

            if action_dict[0] > 0:
                MoveX = 1.0
            elif action_dict[0] == 0:
                MoveX = 0.0
            elif action_dict[0] < 0:
                MoveX = -1.0

            if action_dict[1] > 0:
                Rotate = 1.0
            elif action_dict[1] == 0:
                Rotate = 0.0
            elif action_dict[1] < 0:
                Rotate = -1.0

            if action_dict[2] > 0:
                ChaseFocus = 1.0
            elif action_dict[2] == 0:
                ChaseFocus = 0.0
        else:
            MoveX = 0.0
            Rotate = 0.0
            ChaseFocus = 0.0

            if random.random() <= 0.5:
                CastingSlot = random.randint(1, 2)
                ChangeFocus = 0
            else:
                CastingSlot = 0
                ChangeFocus = random.randint(4, 7)


        #MoveX = 1.0
        #Rotate = 1.0
        #ChaseFocus = 1.0

        #if MoveX != 0 or Rotate != 0 or ChaseFocus != 0:
        #    CastingSlot = 0
        #    ChangeFocus = 0

        action_dict_new = (MoveX, Rotate, ChaseFocus, CastingSlot, ChangeFocus)
        action_index = self.all_actions_dict[action_dict_new]
        #print("action_index: " + str(action_index))

        return action_dict_new, action_index

    @staticmethod
    def dict_to_sorted_str(dict_):
        return str(sorted(dict_.items()))

    def get_key_by_action_dict(self, action_dict):
        for ignored_key in self.ignore_keys:
            action_dict.pop(ignored_key, None)
        str_dict = self.dict_to_sorted_str(action_dict)
        return self.all_actions_dict[str_dict]

    def get_action_dict_by_key(self, key):
        return self.key_to_dict[key]

    def get_actions_dim(self):
        return len(self.key_to_dict)


def get_dtype_dict(env):
    #print("env.action_space: " + str(env.action_space))
    #print("env.action_space[0]: " + str(env.action_space[0]))
    #print("env.action_space[3]: " + str(env.action_space[3]))
    #print("env.action_space[3].shape: " + str(env.action_space[3].shape))

    action_shape = env.action_space.shape
    #rint("action_shape: " + str(action_shape))

    action_shape = action_shape if len(action_shape) > 0 else 1
    action_dtype = env.action_space.dtype
    action_dtype = 'int32' if np.issubdtype(action_dtype, int) else action_dtype
    action_dtype = 'float32' if np.issubdtype(action_dtype, float) else action_dtype
    env_dict = {'action': {'shape': action_shape,
                             'dtype': action_dtype},
                'reward': {'dtype': 'float32'},
                'done': {'dtype': 'bool'}
                }
    for prefix in ('', 'next_'):
        if isinstance(env.observation_space, gym.spaces.Dict):
            for name, space in env.observation_space.spaces.items():
                env_dict[prefix + name] = {'shape': space.shape,
                                             'dtype': space.dtype}
        else:
            env_dict[prefix + 'state'] = {'shape': env.observation_space.shape,
                                          'dtype': env.observation_space.dtype}
    dtype_dict = {key: value['dtype'] for key, value in env_dict.items()}
    dtype_dict.update(weights='float32', indexes='int32')

    return env_dict, dtype_dict