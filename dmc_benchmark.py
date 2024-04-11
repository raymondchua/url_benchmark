DOMAINS = [
    "walker",
    "walkerbigleg",
    "quadruped",
    "jaco",
    "cheetah",
    "cheetahbigleg",
    "humanoid",
    "humanoidsmallleg",
    "humanoidbigleg",
]

CHEETAH_TASKS = [
    "cheetah_run",
    "cheetah_run_backward",
    "cheetah_flip",
    "cheetah_flip_backward",
]

CHEETAHBIGLEG_TASKS = [
    "cheetahbigleg_run",
    "cheetahbigleg_run_backward",
    "cheetahbigleg_flip",
    "cheetahbigleg_flip_backward",
]

WALKER_TASKS = [
    "walker_stand",
    "walker_walk",
    "walker_run",
    "walker_flip",
]

WALKERBIGLEG_TASKS = [
    "walkerbigleg_stand",
    "walkerbigleg_walk",
    "walkerbigleg_run",
    "walkerbigleg_flip",
]

QUADRUPED_TASKS = [
    "quadruped_walk",
    "quadruped_run",
    "quadruped_stand",
    "quadruped_jump",
]

JACO_TASKS = [
    "jaco_reach_top_left",
    "jaco_reach_top_right",
    "jaco_reach_bottom_left",
    "jaco_reach_bottom_right",
]

HUMANOID_TASKS = [
    "humanoid_stand",
    "humanoid_walk",
    "humanoid_run",
]

HUMANOIDSMALLLEG_TASKS = [
    "humanoidsmallleg_stand",
    "humanoidsmallleg_walk",
    "humanoidsmallleg_run",
]

HUMANOIDBIGLEG_TASKS = [
    "humanoidbigleg_stand",
    "humanoidbigleg_walk",
    "humanoidbigleg_run",
]

TASKS = (
    WALKER_TASKS
    + QUADRUPED_TASKS
    + JACO_TASKS
    + CHEETAH_TASKS
    + CHEETAHBIGLEG_TASKS
    + WALKERBIGLEG_TASKS
    + HUMANOID_TASKS
    + HUMANOIDSMALLLEG_TASKS
    + HUMANOIDBIGLEG_TASKS
)

PRIMAL_TASKS = {
    "walker": "walker_stand",
    "jaco": "jaco_reach_top_left",
    "quadruped": "quadruped_walk",
    "cheetah": "cheetah_run",
    "cheetahbigleg": "cheetahbigleg_run",
    "walkerbigleg": "walkerbigleg_stand",
    "humanoid": "humanoid_stand",
    "humanoidsmallleg": "humanoidsmallleg_stand",
    "humanoidbigleg": "humanoidbigleg_stand",
}

# CRL_WALKER_TASKS = ['walker_run', 'walker_flip',]
# CRL_QUADRUPED_TASKS = ['quadruped_run', 'quadruped_jump']
#

"""Walker and Cheetah tasks for CRL experiments."""
CRL_WALKER_TASKS = [
    "walker_run",
    "walkerbigleg_run",
]

CRL_WALKER_DIFF_REWARD_TASKS = [
    "walker_walk",
    "walkerbigleg_run",
]

CRL_CHEETAH_TASKS = [
    "cheetah_run",
    "cheetahbigleg_run",
]

CRL_CHEETAH_DIFF_REWARD_TASKS = [
    "cheetah_run",
    "cheetahbigleg_run_backward",
]

CRL_TASKS_SAME_REWARD = {
    'walker': CRL_WALKER_TASKS,
    'cheetah': CRL_CHEETAH_TASKS,
}

CRL_TASKS_DIFF_REWARD = {
    'walker': CRL_WALKER_DIFF_REWARD_TASKS,
    'cheetah': CRL_CHEETAH_DIFF_REWARD_TASKS,
}


