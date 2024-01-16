import math
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import numpy as np
import pandas as pd
from base_envi import Simulation


class GeneticAlgo:
    def __init__(self, input_layer_shape, input_layer_neurons, number_of_hidden_layers, hidden_layer_neurons,
                 output_layer_neurons, inner_layer_activation, outer_layer_activation):
        # Initialization of genetic algorithm parameters and neural network model
        self.input_layer_shape = input_layer_shape
        self.input_layer_neurons = input_layer_neurons
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons
        self.inner_layer_activation = inner_layer_activation
        self.outer_layer_activation = outer_layer_activation
        self.model = self.create_model()  # Create initial neural network
        self.weights = self.model.layers[0].get_weights()[0]  # Get initial weights
        self.bias = self.model.layers[0].get_weights()[1]  # Get initial biases

    def nn_creation(self, number_of_nn):
        # Generate a list of neural networks with random initial weights
        neural_list = []
        for i in range(0, number_of_nn):
            model = self.create_model()
            neural_list.append(model)
        return neural_list

    def crossing(self, parent_nn1, parent_nn2, number_of_mutations):
        # Perform crossover between two parent neural networks to create a child network
        parent_nn1_weights = parent_nn1.get_weights()[0]
        parent_nn2_weights = parent_nn2.get_weights()[0]
        child_nn_weights=parent_nn1_weights
        # Choose random crossover points
        crossover_points = random.sample(range(0,len(parent_nn1_weights)), number_of_chromosome_cuts)
        for chro in crossover_points:
            child_nn_weights[chro]=parent_nn2_weights[chro]

        child_nn_weights = self.mutation(child_nn_weights, number_of_mutations)
        return child_nn_weights

    def mutation(self, nn_weights_to_mutate, number_of_mutations):
        # Introduce mutations to the weights of a neural network
        max_weight = np.max(nn_weights_to_mutate)
        min_weight = np.min(nn_weights_to_mutate)
        len_weights = len(nn_weights_to_mutate)
        each_list_len = len(nn_weights_to_mutate[0])
        for i in range(0, number_of_mutations):
            nn_weights_to_mutate[random.randint(0, len_weights - 1)][
                random.randint(0, each_list_len - 1)] = random.uniform(
                min_weight, max_weight)
        return nn_weights_to_mutate

    def create_model(self):
        # Create a neural network model using Keras Sequential API
        neural = Sequential()
        neural.add(Dense(self.input_layer_neurons, activation=self.inner_layer_activation,
                         input_shape=self.input_layer_shape))
        for i in range(0, self.number_of_hidden_layers):
            neural.add(Dense(self.hidden_layer_neurons[i], activation=self.inner_layer_activation))

        neural.add(Dense(self.output_layer_neurons, activation=self.outer_layer_activation))
        neural.compile(optimizer='adam', loss='binary_crossentropy', metrics='mse')
        return neural

    def create_offsprings(self, offspring_weights, number_of_offsprings):
        # Create a list of offspring neural networks with the given weights
        offspring_list = []
        for i in range(0, number_of_offsprings):
            child_model = self.create_model()
            child_model.layers[0].set_weights([offspring_weights, self.bias])
            offspring_weights = self.mutation(offspring_weights, 2)
            child_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='mse')
            offspring_list.append(child_model)
        return offspring_list


def fitness_func(health, steps, extra=0):
    # Calculate fitness based on health and steps survived
    # Giving 50 percent weightage to health and 30 percent to steps survived and 20 percent to plant change
    fitness = (0.5 * health) + (0.3 * steps) + (0.2 * extra)
    return fitness


if __name__ == '__main__':
    # Initialize genetic algorithm parameters
    gen = GeneticAlgo((8,), 64, 2, [16, 32], 1, 'relu', 'sigmoid')
    df = pd.read_csv('genetic_res.csv')  # Read data from CSV file into DataFrame
    nn_gen = gen.nn_creation(5)  # Generate initial population of neural networks
    number_of_generations = 301
    number_of_mutations = 2
    number_of_chromosome_cuts = 2
    herbivore_no = 5
    carnivore_no = 10
    plant_no = 90
    rock_no = 60
    herbivore_health = 100
    carnivore_health = 100
    weight_select = [0.90, 0.10]
    for i in range(0, number_of_generations):
        print("Generation: ", i)
        env = Simulation(herbivore_no, carnivore_no, plant_no, rock_no, herbivore_health, carnivore_health,
                         sim_controller="custom", speed=120, available_steps=300)
        herbivore_list, carnivore_list, plant_list, _ = env.get_lists()
        herbi1, herbi2, herbi3, herbi4, herbi5 = herbivore_list[0], herbivore_list[1], herbivore_list[2], \
                                                 herbivore_list[3], \
                                                 herbivore_list[4]
        done = 0
        obs_list = []
        obs = 0
        for o in range(0, herbivore_no):
            ob = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            ob = ob.reshape((-1, 8))
            obs_list.append(ob)
        fitness_list = [0] * herbivore_no
        herbi1_x, herbi2_x, herbi3_x, herbi4_x, herbi5_x = [obs_list[0]], [obs_list[0]], [obs_list[0]], [obs_list[0]], [obs_list[0]]
        herbi1_y, herbi2_y, herbi3_y, herbi4_y, herbi5_y = [], [], [], [], []
        plant_no_counter = len(plant_list)
        while done == 0:
            # print(nn_gen[0].predict(obs_list[0])[0])
            action = random.choices([math.ceil(nn_gen[0].predict(obs_list[0])[0][0] * 4), random.randint(1, 4)],
                                    weights=weight_select)[0]
            # print(action)
            done, obs = env.step(herbi1, action)
            if type(obs) == list:
                obs_list[0] = np.array(obs[0:8]).reshape((-1, 8))
                herbi1_x.append(obs_list[0])
                herbi1_y.append(action)
                fitness_list[0] = fitness_func(obs[-2]-herbivore_health, obs[-1],plant_no_counter-len(plant_list))
                plant_no_counter=len(plant_list)
            action = random.choices([math.ceil(nn_gen[1].predict(obs_list[1])[0][0] * 4), random.randint(1, 4)],
                                    weights=weight_select)[0]
            done, obs = env.step(herbi2, action)
            if type(obs) == list:
                obs_list[1] = np.array(obs[0:8]).reshape((-1, 8))
                herbi2_x.append(obs_list[1])
                herbi2_y.append(action)
                fitness_list[1] = fitness_func(obs[-2]-herbivore_health, obs[-1], plant_no_counter - len(plant_list))
                plant_no_counter = len(plant_list)
            action = random.choices([math.ceil(nn_gen[2].predict(obs_list[2])[0][0] * 4), random.randint(1, 4)],
                                    weights=weight_select)[0]
            done, obs = env.step(herbi3, action)
            if type(obs) == list:
                obs_list[2] = np.array(obs[0:8]).reshape((-1, 8))
                herbi3_x.append(obs_list[2])
                herbi3_y.append(action)
                fitness_list[2] = fitness_func(obs[-2]-herbivore_health, obs[-1], plant_no_counter - len(plant_list))
                plant_no_counter = len(plant_list)
            action = random.choices([math.ceil(nn_gen[3].predict(obs_list[3])[0][0] * 4), random.randint(1, 4)],
                                    weights=weight_select)[0]
            done, obs = env.step(herbi4, action)
            if type(obs) == list:
                obs_list[3] = np.array(obs[0:8]).reshape((-1, 8))
                herbi4_x.append(obs_list[3])
                herbi4_y.append(action)
                fitness_list[3] = fitness_func(obs[-2]-herbivore_health, obs[-1], plant_no_counter - len(plant_list))
                plant_no_counter = len(plant_list)
            action = random.choices([math.ceil(nn_gen[4].predict(obs_list[4])[0][0] * 4), random.randint(1, 4)],
                                    weights=weight_select)[0]
            done, obs = env.step(herbi5, action)
            if type(obs) == list:
                obs_list[4] = np.array(obs[0:8]).reshape((-1, 8))
                herbi5_x.append(obs_list[4])
                herbi5_y.append(action)
                fitness_list[4] = fitness_func(obs[-2]-herbivore_health, obs[-1], plant_no_counter - len(plant_list))
                plant_no_counter = len(plant_list)
            for carni in carnivore_list[-2::-1]:
                done, obs = env.step(carni, random.randint(1, 4))
            plant_no_counter = len(plant_list)
            # print("Fitness: ", fitness_list)
        herbi1_x.pop()
        herbi2_x.pop()
        herbi3_x.pop()
        herbi4_x.pop()
        herbi5_x.pop()
        total_x = [herbi1_x, herbi2_x, herbi3_x, herbi4_x, herbi5_x]
        total_y = [herbi1_y, herbi2_y, herbi3_y, herbi4_y, herbi5_y]
        # print(total_x)
        # print(total_y)
        dic_res = {'Generation': i, 'Winner': obs, "Fitness": fitness_list}
        df = df._append(dic_res, ignore_index=True)  # Append results to DataFrame
        df.to_csv('genetic_res.csv', index=False)  # Save DataFrame to CSV
        # print(fitness_list)
        max_index = fitness_list.index(max(fitness_list))
        # print(fitness_list[max_index])
        max1=fitness_list[max_index]
        fitness_list[max_index] = -100
        max_index_2 = fitness_list.index(max(fitness_list))
        # print(fitness_list[max_index_2])
        fitness_list[max_index]=max1
        parent1 = nn_gen[max_index]
        parent1.fit(np.array(total_x[max_index]).reshape(-1, 8), np.array(total_y[max_index]), epochs=20, verbose=0)
        parent2 = nn_gen[max_index_2]
        parent2.fit(np.array(total_x[max_index_2]).reshape(-1, 8), np.array(total_y[max_index_2]), epochs=20, verbose=0)
        child_weights = gen.crossing(parent1, parent2, number_of_mutations)
        nn_gen = gen.create_offsprings(child_weights, herbivore_no)
        if i % 10 == 0:
            nn_gen[0].save(f"models/gen{i}_model.h5")
    print("Genetic Search DONE !")
