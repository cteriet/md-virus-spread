import numpy as np
import random
import json
import codecs
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import cv2
import glob
import re
from agent import Agent


class System:
    def __init__(self, parameters):
        # Parameters
        self.DT = parameters['DT']  # 10e-3
        self.MAXSTEP = parameters['MAXSTEP']  # 10000

        # Norms, Geometry, Force Function, Disease Function
        self.norm = parameters['norm']
        self.vector_difference = parameters['vector_difference']
        self.boundary_condition = parameters['boundary_condition']
        self.energy_drift_compensation = parameters['energy_drift_compensation']
        self.energy_drift_compensation_slope = parameters['energy_drift_compensation_slope']
        self.energy_drift_compensation_vmax = parameters['energy_drift_compensation_vmax']
        self.energy_drift_compensation_clipspeed = parameters['energy_drift_compensation_clipspeed']
        self.force = parameters['force']
        self.force_constant = parameters['force_constant']
        self.box = parameters['box']

        # Dynamic System Variables
        self.time = 0
        self.agents = []

        # Measurements
        self.write_interval = parameters['write_interval']
        self.measurements = {}

        # Color palette
        self.agent_type_colors = parameters['agent_type_colors']
        self.agent_status_colors = parameters['agent_status_colors']

        # Image and Video Export
        self.export_path = parameters['export_path']
        self.measurements_file = parameters['measurements_file']
        self.image_export_name = parameters['image_export_name']
        self.image_export_format = parameters['image_export_format']
        self.video_export_name = parameters['video_export_name']
        self.video_export_format = parameters['video_export_format']
        self.video_export_fps = parameters['video_export_fps']

    def __str__(self):
        return "System contains " + str(len(self.agents)) + " agents at time " + str(self.time)

    # ------------------------------------------------------------------------------------------------------------------
    # Handle System Plotting, Saving, Styling
    # ------------------------------------------------------------------------------------------------------------------
    def agent_status_color(self, agent_status):
        return self.agent_status_colors[agent_status]

    def agent_type_color(self, agent_type):
        return self.agent_type_colors[agent_type]

    @staticmethod
    def status_to_array(status):
        status_array = np.array([0, 0, 0, 0, 0])
        status_array[status] = 1

        return status_array

    def measure(self):
        measurements = {}

        for agent in self.agents:
            if agent.type in measurements:
                measurements[agent.type] = measurements[agent.type] + self.status_to_array(agent.state)
            else:
                measurements[agent.type] = self.status_to_array(agent.state)

        for key in list(measurements.keys()):
            measurements[key] = np.append(measurements[key], self.time)
            if key in self.measurements:
                self.measurements[key].append(measurements[key].tolist())
            else:
                self.measurements[key] = [measurements[key].tolist()]

    def write(self):
        json.dump(self.measurements, codecs.open(self.measurements_file, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

    def save_plot(self, export_path):
        agent_positions = [agent.position for agent in self.agents]
        agent_size = [2*agent.size for agent in self.agents]
        agent_type_colors = [self.agent_type_color(agent.type) for agent in self.agents]
        agent_status_colors = [self.agent_status_color(agent.state) for agent in self.agents]

        fig, ax = plt.subplots()
        ax.grid(True)
        ax.axis(xmin=-25, xmax=self.box[0] + 25, ymin=-25, ymax=self.box[1] + 25)
        ax.set_aspect(1)

        points = EllipseCollection(widths=agent_size,
                                   heights=agent_size,
                                   angles=0,
                                   units='xy',
                                   linewidths=2,
                                   transOffset=ax.transData,
                                   alpha=0.3,
                                   facecolors=agent_type_colors,
                                   edgecolors=agent_status_colors,
                                   offsets=agent_positions)

        ax.add_collection(points)

        fig.savefig(export_path, dpi=300)
        plt.close()

    def create_animation_from_folder(self):
        img_array = []
        filenames = glob.glob(self.export_path + self.image_export_name + '*.' + self.image_export_format)
        filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

        if len(filenames) > 0:
            for filename in filenames:
                img = cv2.imread(filename)
                img_array.append(img)

            height, width, layers = img_array[0].shape
            video_size = (width, height)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(self.export_path + self.video_export_name + '.' + self.video_export_format, fourcc, self.video_export_fps, video_size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

    # ------------------------------------------------------------------------------------------------------------------
    # Handle System Simulation
    # ------------------------------------------------------------------------------------------------------------------
    def apply_boundary_conditions(self):
        for agent in self.agents:
            agent_position, agent_velocity = self.boundary_condition(agent.position, agent.velocity, self.box)

            agent.set_position(agent_position)
            agent.set_velocity(agent_velocity)

    def add_agent(self, parameters):
        overlap = True
        while overlap:
            overlap = False
            new_agent = Agent(parameters)
            for agent in self.agents:
                if self.norm(new_agent.position, agent.position, self.box) <= (new_agent.size + agent.size):
                    print('Overlap: ', new_agent.position, ' and ', agent.position)
                    overlap = True
                    break

        self.agents.append(new_agent)

    def handle_force(self, agent_position, other_agent_position):
        vec = self.vector_difference(other_agent_position, agent_position, self.box)
        vec = vec / np.linalg.norm(vec)

        r = self.norm(agent_position, other_agent_position, self.box)

        return self.force(r, self.force_constant) * vec

    def step(self):
        for index, agent in enumerate(self.agents):
            # Progress the agent disease state
            agent.handle_state()

            # Handle the forces applied to the agent by other agents, and check if other agent get infected
            for other_agent in [x for i, x in enumerate(self.agents) if i != index]:

                # Handle possible infection of other agents from agent
                if other_agent.state == 0 and (agent.state == 1 or agent.state == 2):
                    other_agent.get_infection(agent.disease_profile(self.norm(agent.position, other_agent.position, self.box)) * other_agent.infection_profile(self.norm(agent.position, other_agent.position, self.box)))

                # Handle Forces applied on agent by other_agents
                agent.add_force(self.handle_force(agent.position, other_agent.position))

            agent.set_velocity(self.energy_drift_compensation(agent.velocity, self.energy_drift_compensation_slope, self.energy_drift_compensation_vmax, self.energy_drift_compensation_clipspeed))
            # Move the agent
            agent.move()

        self.apply_boundary_conditions()
        random.shuffle(self.agents)

    def run(self):
        for i in range(self.MAXSTEP):
            self.save_plot(self.export_path + self.image_export_name + str(i) + '.' + self.image_export_format)

            self.step()

            if i % 10 == 0:
                print('Step: ', i)

            if i % self.write_interval == 0:
                self.measure()

            self.time = self.time + 1

        self.write()

