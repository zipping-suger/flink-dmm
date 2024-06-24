import json

# - ending world2box transformation
def load_composition(composition):
    return composition["moveBoxes"][0]["targets"][0]["world2box"]["pos"]

# - robot initial angles
def load_robot(robot):
    return robot["state"]["angles"]

# - initial world2box transformation
def load_space(space):
    return space["boxes"]["box0"]["world2box"]["pos"]

# - series of joint angles
def load_memory(memory):
    return memory[1]["pn"]

