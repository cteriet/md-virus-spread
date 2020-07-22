'''
Created on June 22, 2020
@author: Christian te Riet
'''

# !/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from system import System


def main():

    # System Norms & Boundary Conditions
    def vector_difference(vec1, vec2, box):
        x1 = (vec2[0] - vec1[0]) % box[0]
        x2 = (vec1[0] - vec2[0]) % box[0]

        y1 = (vec2[1] - vec1[1]) % box[1]
        y2 = (vec1[1] - vec2[1]) % box[1]

        if x1 < x2:
            x = x1
        else:
            x = -x2

        if y1 < y2:
            y = y1
        else:
            y = -y2

        return np.array([x, y])

    def norm(vec1, vec2, box, minimum=0):
        return max(np.linalg.norm(vector_difference(vec1, vec2, box)), minimum)

    def boundary_condition(position, velocity, box):
        for index, coordinate in enumerate(position):
            position[index] = position[index] % box[index]

        return position, velocity

    def force(r, c=10):
        return -12*c*np.power(r+0.1, -13) + 6*c*np.power(r+0.1, -7)

    def energy_drift_compensation(v, s=0.25, vmax=1, clipspeed=1000):
        v_norm = np.linalg.norm(v)
        v_clipped = min(v_norm, clipspeed)
        return (v/v_norm)*vmax*(1/(1+np.exp(-s*v_clipped)) - 1/2)

    def disease_profile(r, r0, p):
        if r < r0:
            return p
        else:
            return 0.0

    def infection_profile(r, r0, p):
        if r < r0:
            return p
        else:
            return 0.0

    system_params = {
        'DT': 1e0,
        'MAXSTEP': 400,
        'box': np.array([125, 125]),
        'write_interval': 10,
        'agent_type_colors': {
            'Healthy': "#49BA50",
            'Initial_sick': '#49BA50',
            'Old': '#BA3296',
            'Young': '#4496BA'
        },
        'agent_status_colors': ['#007AB2', '#B29D00', '#B22E00', '#13C000', '#5B5B5B'],
        'export_path': './healthy_old_young/',                                          # export folder (this folder has to exist/isn't created automatically)
        'measurements_file': 'healthy_old_young.json',                                  # name of file where experiment results are stored
        'image_export_name': '',
        'image_export_format': 'png',
        'video_export_name': 'healthy_old_young',
        'video_export_format': 'avi',
        'video_export_fps': 15,

        'norm': norm,
        'vector_difference': vector_difference,
        'boundary_condition': boundary_condition,

        'force': force,
        'force_constant': 1000,

        'energy_drift_compensation': energy_drift_compensation,
        'energy_drift_compensation_slope': 1,
        'energy_drift_compensation_vmax': 5,
        'energy_drift_compensation_clipspeed': 1000
    }

    healthy_agent_parameters = {
        'status': 0,
        'immobile': False,
        'transparent': False,
        'type': 'Healthy',
        'size': 2,
        'mass': 1,
        'DT': system_params['DT'],
        'healthy_velocity': 1,
        'incubation_velocity': 1,
        'sickness_velocity': 1,
        'recoverProbability': 1.0,
        'timeToRecover': lambda: int(np.random.normal(80, 10)),
        'timeToDie': lambda: int(np.random.normal(50, 10)),
        'timeToIncubate': lambda: int(np.random.normal(50, 15)),
        'box': system_params['box'],
        'disease_profile': lambda r: disease_profile(r, 4, 0.75),
        'infection_profile': lambda r: infection_profile(r, 4, 0.75)
    }

    sick_agent_parameters = {
        'status': 2,
        'immobile': False,
        'transparent': False,
        'type': 'Healthy',
        'size': 2,
        'mass': 1,
        'DT': system_params['DT'],
        'healthy_velocity': 1,
        'incubation_velocity': 1,
        'sickness_velocity': 1,
        'recoverProbability': 0.95,
        'timeToRecover': lambda: int(np.random.normal(80, 10)),
        'timeToDie': lambda: int(np.random.normal(50, 10)),
        'timeToIncubate': lambda: int(np.random.normal(50, 15)),
        'box': system_params['box'],
        'disease_profile': lambda r: disease_profile(r, 4, 0.70),
        'infection_profile': lambda r: infection_profile(r, 4, 0.70)
    }

    old_agent_parameters = {
        'status': 0,
        'immobile': False,
        'transparent': False,
        'type': 'Old',
        'size': 2,
        'mass': 1,
        'DT': system_params['DT'],
        'healthy_velocity': 1,
        'incubation_velocity': 1,
        'sickness_velocity': 1,
        'recoverProbability': 0.80,
        'timeToRecover': lambda: int(np.random.normal(80, 10)),
        'timeToDie': lambda: int(np.random.normal(50, 10)),
        'timeToIncubate': lambda: int(np.random.normal(50, 15)),
        'box': system_params['box'],
        'disease_profile': lambda r: disease_profile(r, 4, 0.90),
        'infection_profile': lambda r: infection_profile(r, 4, 0.70)
    }

    young_agent_parameters = {
        'status': 0,
        'immobile': False,
        'transparent': False,
        'type': 'Young',
        'size': 2,
        'mass': 1,
        'DT': system_params['DT'],
        'healthy_velocity': 1,
        'incubation_velocity': 1,
        'sickness_velocity': 1,
        'recoverProbability': 0.95,
        'timeToRecover': lambda: int(np.random.normal(80, 10)),
        'timeToDie': lambda: int(np.random.normal(50, 10)),
        'timeToIncubate': lambda: int(np.random.normal(50, 15)),
        'box': system_params['box'],
        'disease_profile': lambda r: disease_profile(r, 4, 0.40),
        'infection_profile': lambda r: infection_profile(r, 4, 0.70)
    }

    system = System(system_params)

    print('Adding Healthy Agents...')
    for i in range(56):
        system.add_agent(healthy_agent_parameters)

    print('Adding Sick Agents...')
    for i in range(4):
        system.add_agent(sick_agent_parameters)

    print('Adding Old Agents...')
    for i in range(20):
        system.add_agent(old_agent_parameters)

    print('Adding Young Agents...')
    for i in range(20):
        system.add_agent(young_agent_parameters)

    print('Running Simulations')
    system.run()

    print('Creating Video...')
    system.create_animation_from_folder()


if __name__ == '__main__':
    main()
