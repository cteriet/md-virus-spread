# md-virus-spread
Agent based Molecular Dynamics-esque simulation of virus spread in a population of agents

I tried making a small python simulation analogous to this simulation and article in the Washington Post: https://www.washingtonpost.com/graphics/2020/world/corona-simulator/

The simulation program has two classes: the agent and the system itself. The agent class contains all code relevant to agent behaviour and interaction, such as movement speed, recovery likelihood and what happens during an collision between agents. The system class contains settings and measurementresults of the environment wherein the agents live, such as the spatial dimensions and code related to creating snapshots of the simulation.

I "cheated" a bit in the simulation. I wanted to keep the number of steps needed to run the simulation low so that the simulation runs quickly. A small number of steps, requires a relatively large step size which can introduce numerical instabillity (i.e. agents suddenly very closely overlapping, resulting in a "unphysical" increase in the total energy of the simulated system) due to numerical instabilities as a result of too large stepsizes in the simulation. To overcome this problem, I introduced a soft "speed limit" which prevents agents from moving faster than a certain limit. This speed limit might not be quite realistic, but since the goal was to just let agents interact and mix with eachother, I thought it wasn't too big of a deal. A more realistic solution would be to use some function based in physics to account for particle interactions (for instance, a Lennard-Jones potential), using C/cython, and decreasing the step size. 

## Running the simulation
To run the simulation, see corona_simulation.py and run:

```
python corona_simulation.py
```

In corona_simulation.py, you can change the parameters of the simulation
