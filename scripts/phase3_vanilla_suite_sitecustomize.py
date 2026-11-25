"""Force vanilla 10-task libero_spatial into libero's benchmark registry."""
from pathlib import Path
TASKS=[
'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate',
'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate',
'pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate',
'pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate',
'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate',
'pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate',
'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate',
'pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate',
'pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate',
'pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate',
]
try:
 import libero.libero.benchmark as b
 tasks=[b.Task(name=t,language=b.grab_language_from_filename('libero_spatial',t+'.bddl'),problem='Libero',problem_folder='libero_spatial',bddl_file=t+'.bddl',init_states_file=t+'.pruned_init') for t in TASKS]
 class VanillaSpatial(b.Benchmark):
  name='libero_spatial'
  def __init__(self,*a,**k): super().__init__(*a,**k); self.tasks=tasks; self.n_tasks=len(tasks)
  def _make_benchmark(self): pass
 b.BENCHMARK_MAPPING['libero_spatial']=VanillaSpatial
except Exception as e:
 import sys; print('sitecustomize patch failed',e,file=sys.stderr)
