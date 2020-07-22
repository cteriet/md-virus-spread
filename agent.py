import numpy as np


class Counter:
    def __init__(self, tmax):
        self.tmax = tmax
        self.time = 1
        self.expired = False

    def reset(self):
        self.time = 1
        self.expired = False

    def step(self):
        if self.time < self.tmax:
            self.time += 1
        else:
            self.expired = True


class Agent:
    def __init__(self, parameters):
        # Parameters
        self.type = parameters['type']
        self.size = parameters['size'] if not callable(parameters['size']) else parameters['size']()
        self.mass = parameters['mass'] if not callable(parameters['mass']) else parameters['mass']()
        self.DT = parameters['DT']
        self.willRecover = (np.random.uniform(0, 1) < parameters['recoverProbability'])

        # Bounding box
        self.box = parameters['box']

        # Velocities
        self.healthy_velocity = parameters['healthy_velocity'] if not callable(parameters['healthy_velocity']) else parameters['healthy_velocity']()
        self.sickness_velocity = parameters['sickness_velocity'] if not callable(parameters['sickness_velocity']) else parameters['sickness_velocity']()
        self.incubation_velocity = parameters['incubation_velocity'] if not callable(parameters['sickness_velocity']) else parameters['sickness_velocity']()

        # Infection Rates
        self.disease_profile = parameters['disease_profile'] if not callable(parameters['disease_profile']) else parameters['disease_profile']
        self.infection_profile = parameters['infection_profile'] if not callable(parameters['infection_profile']) else parameters['infection_profile']

        # Illness and Recover Rates
        self.timeToRecover = parameters['timeToRecover'] if not callable(parameters['timeToRecover']) else parameters['timeToRecover']()
        self.timeToDie = parameters['timeToDie'] if not callable(parameters['timeToDie']) else parameters['timeToDie']()
        self.timeToIncubate = parameters['timeToIncubate'] if not callable(parameters['timeToIncubate']) else parameters['timeToIncubate']()

        # Dynamic Variables
        self.state = parameters['status'] if not callable(parameters['status']) else parameters['status']()
        self.immobile = parameters['immobile'] if not callable(parameters['immobile']) else parameters['immobile']()
        self.transparent = parameters['transparent'] if not callable(parameters['transparent']) else parameters['transparent']()
        self.position = self.set_initial_position()
        self.velocity = self.set_initial_velocity()

        # Counter
        self.set_counter_status()

    def set_initial_position(self):
        pos = np.random.rand(1, 2)[0]
        pos[0] = pos[0]*self.box[0]
        pos[1] = pos[1]*self.box[1]

        return pos

    def set_initial_velocity(self):
        vel = 2*(np.random.rand(1, 2)[0] - 0.5)
        vel = vel / np.linalg.norm(vel)

        if self.state == 0:
            vel = self.healthy_velocity * vel
        elif self.state == 1:
            vel = self.incubation_velocity * vel
        elif self.state == 2:
            vel = self.sickness_velocity * vel
        elif self.state == 3:
            vel = self.healthy_velocity * vel
        else:
            vel = 0

        return vel

    def set_position(self, position):
        self.position = position

    def set_state(self, state):
        self.state = state

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_mass(self, mass):
        self.mass = mass

    def set_velocity_magnitude(self, velocity_magnitude):
        self.velocity = velocity_magnitude*self.velocity/np.linalg.norm(self.velocity)

    def add_force(self, force):
        self.velocity = self.velocity + (force/self.mass)*self.DT

    def move(self):
        self.set_position(self.position + self.velocity * self.DT)

    def get_infection(self, probability):
        if np.random.uniform(0, 1) < probability:
            self.enter_incubation_phase()

    def set_counter_status(self):
        if self.state == 1:
            self.counter = Counter(self.timeToIncubate)
        elif self.state == 2:
            self.counter = Counter(self.timeToRecover) if self.willRecover else Counter(self.timeToDie)
        else:
            self.counter = None

    def enter_incubation_phase(self):
        self.set_state(1)
        self.set_velocity_magnitude(self.incubation_velocity)
        self.set_counter_status()

    def become_sick(self):
        self.set_state(2)
        self.set_velocity_magnitude(self.sickness_velocity)
        self.set_counter_status()

    def death(self):
        self.transparent = True
        self.set_velocity_magnitude(0)
        self.immobile = True
        self.set_state(4)
        self.set_counter_status()

    def recover(self):
        self.set_state(3)
        self.set_velocity_magnitude(self.healthy_velocity)
        self.set_counter_status()

    def handle_state(self):
        if self.state == 1 or self.state == 2:
            self.counter.step()

            if self.counter.expired:
                if self.state == 1:
                    self.become_sick()
                elif self.state == 2 and self.willRecover:
                    self.recover()
                elif self.state == 2 and not self.willRecover:
                    self.death()

    # todo lennard jones
    # def forces(self, agent):
    #     dx = self.norm(self.position, agent.position, self.size + agent.size)
    #
    #     if agent.immobile:
    #         v_self = self.velocity - 2 * self.vector_difference(self.position, agent.position) * np.dot(self.velocity, self.vector_difference(self.position, agent.position)) / (dx*dx)
    #         agent.set_velocity(v_self)
    #     else:
    #         v_self = self.velocity - 2 * (agent.mass/(self.mass + agent.mass)) * self.vector_difference(self.position, agent.position) * np.dot(self.velocity - agent.velocity, self.vector_difference(self.position, agent.position)) / (dx*dx)
    #         v_agent = agent.velocity - 2 * (self.mass/(agent.mass + self.mass)) * self.vector_difference(agent.position, self.position) * np.dot(agent.velocity - self.velocity, self.vector_difference(agent.position, self.position)) / (dx*dx)
    #
    #         self.set_velocity(v_self)
    #         agent.set_velocity(v_agent)
    #
    # def handle_forces(self, agent):
    #     if not self.transparent and self.vector_difference(self.position, agent.position) < (self.size + agent.size):
    #         # Collision Event
    #         self.forces(agent)
    #
    #         # Infection Event
    #         self.handle_infection(agent)
