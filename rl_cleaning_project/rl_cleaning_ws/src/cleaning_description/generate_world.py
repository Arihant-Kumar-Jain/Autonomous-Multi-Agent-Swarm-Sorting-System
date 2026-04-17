import numpy as np

def generate():
    grid_size = 50
    cell_size = 0.2
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid.fill(-1)
    for i in range(grid_size):
        if i % 5 == 0 or i % 5 == 1:
            grid[i, :] = 0
            grid[:, i] = 0
            
    # generate sdf
    sdf = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <model name="maze">
      <static>true</static>
'''
    wall_idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == -1:
                x = (c - grid_size/2.0) * cell_size + cell_size/2.0
                y = (grid_size/2.0 - r) * cell_size - cell_size/2.0
                sdf += f'''
      <link name="wall_{wall_idx}">
        <pose>{x} {y} 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{cell_size} {cell_size} 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{cell_size} {cell_size} 0.2</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>'''
                wall_idx += 1
                
    sdf += '''
    </model>
  </world>
</sdf>'''

    with open('/home/aman/cs671_7/rl_cleaning_project/rl_cleaning_ws/src/cleaning_description/worlds/simple_roads.world', 'w') as f:
        f.write(sdf)

if __name__ == '__main__':
    generate()
