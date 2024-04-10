from custom_dmc_tasks import cheetah
from custom_dmc_tasks import cheetahbigleg
from custom_dmc_tasks import walker
from custom_dmc_tasks import walkerbigleg
from custom_dmc_tasks import hopper
from custom_dmc_tasks import quadruped
from custom_dmc_tasks import jaco
from custom_dmc_tasks import humanoid
from custom_dmc_tasks import humanoidsmallleg
from custom_dmc_tasks import humanoidbigleg


def make(
    domain, task, task_kwargs=None, environment_kwargs=None, visualize_reward=False
):

    if domain == "cheetah":
        return cheetah.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "cheetahbigleg":
        return cheetahbigleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "walker":
        return walker.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "walkerbigleg":
        return walkerbigleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "hopper":
        return hopper.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "quadruped":
        return quadruped.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "humanoid":
        return humanoid.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "humanoidsmallleg":
        return humanoidsmallleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    elif domain == "humanoidbigleg":
        return humanoidbigleg.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )

    else:
        raise f"{task} not found"

    assert None


def make_jaco(task, obs_type, seed):
    return jaco.make(task, obs_type, seed)
