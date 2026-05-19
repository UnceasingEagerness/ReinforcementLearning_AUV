import holoocean
from multiagent.multi_asv_scenario import SCENARIO

env = holoocean.make(
    scenario_cfg=SCENARIO,
    show_viewport=True
)

while True:

    env.act("vessel0", [50000, 50000])

    state = env.tick()

    dyn = state["vessel0"]["DynamicsSensor"]

    print(dyn[6:9])