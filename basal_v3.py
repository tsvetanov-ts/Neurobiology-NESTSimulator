#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest
import matplotlib.pyplot as plt
import numpy as np
import random
# 1. Нулиране на ядрото и задаване на времето на симулация
nest.ResetKernel()
simulation_time = 1000.0  # симулация за 1000 ms

# 2. Дефиниране на типовете неврони и броя им
neuron_types = ["STN", "GPe", "MSd", "MSi", "FSd", "FSi", "SNr", "SC"]
num_neurons = {
    "STN": 25,
    "GPe": 25,
    "MSd": 50,
    "MSi": 50,
    "FSd": 5,
    "FSi": 5,
    "SNr": 25,
    "SC": 10
}

# 3. Създаване на популациите за Лявата (LEFT) и Дясната (RIGHT) половина
LEFT = {}
RIGHT = {}
for n_type in neuron_types:
    LEFT[n_type] = nest.Create("iaf_psc_alpha", num_neurons[n_type])
    RIGHT[n_type] = nest.Create("iaf_psc_alpha", num_neurons[n_type])
    # При нужда могат да се подадат специфични параметри чрез параметъра params

# 4. Подаване на силен вход – комбинираме Poisson генератори и DC генератори.
#
# Функция за добавяне на Poisson шумов вход:
# def add_poisson_input(pop_dict, rate=2000.0, weight=2.0, delay=1.0):
#     poisson_dict = {}
#     for n_type, pop in pop_dict.items():
#         noise = nest.Create("poisson_generator", params={"rate": rate})
#         poisson_dict[n_type] = noise
#         nest.Connect(noise, pop, syn_spec={"weight": weight, "delay": delay})
#     return poisson_dict
#
# # Функция за добавяне на постоянен DC вход:
# def add_dc_input(pop_dict, amplitude=350.0):
#     dc_dict = {}
#     for n_type, pop in pop_dict.items():
#         dc = nest.Create("dc_generator", params={"amplitude": amplitude})
#         dc_dict[n_type] = dc
#         nest.Connect(dc, pop)
#     return dc_dict

def add_poisson_input(pop_dict, rate=2000.0, weight=2.0, delay=1.0, shuffle=True,
                      rate_range=(1900.0, 2100.0), weight_range=(1.8, 2.2), delay_range=(0.9, 1.1)):
    """
    Добавя Poisson входове към всяка популация.
    Ако shuffle=True, за всяка популация се генерират произволни стойности на rate, weight и delay,
    избрани в зададените интервали.

    :param pop_dict: Речник със популации (ключове – имена на популациите).
    :param rate: Стойност за rate, ако shuffle=False.
    :param weight: Стойност за weight, ако shuffle=False.
    :param delay: Стойност за delay, ако shuffle=False.
    :param shuffle: Ако е True, се генерират произволни параметри.
    :param rate_range: Интервал за произволен rate (min, max).
    :param weight_range: Интервал за произволен weight (min, max).
    :param delay_range: Интервал за произволен delay (min, max).
    :return: Речник с Poisson генератори за всяка популация.
    """
    poisson_dict = {}
    for n_type, pop in pop_dict.items():
        if shuffle:
            r = random.uniform(*rate_range)
            w = random.uniform(*weight_range)
            d = random.uniform(*delay_range)
        else:
            r = rate
            w = weight
            d = delay
        noise = nest.Create("poisson_generator", params={"rate": r})
        poisson_dict[n_type] = noise
        nest.Connect(noise, pop, syn_spec={"weight": w, "delay": d})
    return poisson_dict


# Функция за DC вход с рандомизация
def add_dc_input(pop_dict, amplitude=350.0, shuffle=True, amplitude_range=(330.0, 370.0)):
    """
    Добавя DC входове към всяка популация.
    Ако shuffle=True, за всяка популация се генерира произволна амплитуда в дадения интервал.

    :param pop_dict: Речник със популации.
    :param amplitude: Стойност за амплитуда, ако shuffle=False.
    :param shuffle: Ако е True, се генерира произволна амплитуда.
    :param amplitude_range: Интервал за произволна амплитуда (min, max).
    :return: Речник с DC генератори за всяка популация.
    """
    dc_dict = {}
    for n_type, pop in pop_dict.items():
        if shuffle:
            amp = random.uniform(*amplitude_range)
        else:
            amp = amplitude
        dc = nest.Create("dc_generator", params={"amplitude": amp})
        dc_dict[n_type] = dc
        nest.Connect(dc, pop)
    return dc_dict

# Добавяме към Лявата и Дясната половина
noise_LEFT = add_poisson_input(LEFT)
noise_RIGHT = add_poisson_input(RIGHT)
dc_LEFT = add_dc_input(LEFT)
dc_RIGHT = add_dc_input(RIGHT)

# 5. Създаване на spike_recorders за всяка популация (от Лявата и Дясната половина)
spike_recorders_LEFT = {}
spike_recorders_RIGHT = {}
for n_type in neuron_types:
    sr_left = nest.Create("spike_recorder")
    sr_right = nest.Create("spike_recorder")
    spike_recorders_LEFT[n_type] = sr_left
    spike_recorders_RIGHT[n_type] = sr_right
    nest.Connect(LEFT[n_type], sr_left)
    nest.Connect(RIGHT[n_type], sr_right)

# 6. Дефиниране на синаптичните връзки за всяка популация
#
# Тук дефинираме „индивидуални“ синаптични връзки – за всеки източник задаваме списък от
# връзки към целеви популации с определени параметри.
#
# В този пример:
# - Изходът на STN и SC (възбудни) използва положително тегло (+3.0)
# - Изходът на останалите (GPe, MSd, MSi, FSd, FSi, SNr – инхибиторни) използва отрицателно тегло (около -3.0)
#
connections_left = {
    "STN": [
        ("GPe", 3.0, 1.0, "all_to_all"),
        ("SC", 3.0, 1.0, "all_to_all")
    ],
    "GPe": [
        ("STN", -3.0, 1.0, "all_to_all"),
        ("MSd", -2.5, 1.5, "pairwise_bernoulli", 0.3),
        ("MSi", -2.5, 1.5, "pairwise_bernoulli", 0.3)
    ],
    "MSd": [
        ("FSd", -3.0, 1.0, "all_to_all"),
        ("SNr", -3.0, 1.0, "all_to_all")
    ],
    "MSi": [
        ("FSi", -3.0, 1.0, "all_to_all"),
        ("SNr", -3.0, 1.0, "all_to_all")
    ],
    "FSd": [
        ("GPe", -2.0, 1.0, "all_to_all"),
        ("SNr", -3.0, 1.0, "all_to_all")
    ],
    "FSi": [
        ("GPe", -2.0, 1.0, "all_to_all"),
        ("SNr", -3.0, 1.0, "all_to_all")
    ],
    "SNr": [
        ("STN", -3.0, 1.0, "all_to_all"),
        ("SC", -3.0, 1.0, "all_to_all")
    ],
    "SC": [
        ("STN", 3.0, 1.0, "all_to_all"),
        ("GPe", 3.0, 1.0, "all_to_all")
    ]
}
# За дясната половина използваме идентична схема
connections_right = connections_left.copy()

# Функция за свързване на връзките за дадена половина:
def connect_population(pop_dict, connection_dict):
    for source, conns in connection_dict.items():
        for conn in conns:
            # Ако връзката съдържа 4 елемента: (target, weight, delay, rule)
            # Ако съдържа 5 елемента: (target, weight, delay, rule, p) → за pairwise_bernoulli
            if len(conn) == 4:
                target, weight, delay, rule = conn
                conn_dict = {"rule": rule}
                syn_spec = {"weight": weight, "delay": delay}
            elif len(conn) == 5:
                target, weight, delay, rule, p = conn
                conn_dict = {"rule": rule, "p": p}
                syn_spec = {"weight": weight, "delay": delay}
            else:
                raise ValueError("Невалиден формат на връзката!")
            nest.Connect(pop_dict[source], pop_dict[target], conn_dict, syn_spec)

connect_population(LEFT, connections_left)
connect_population(RIGHT, connections_right)

# 6.1. Междуполовни (cross) връзки – свързваме съответстващите популации от Лявата и Дясната половина
cross_connections = {
    "STN": [("STN", 3.0, 1.0, "all_to_all")],
    "GPe": [("GPe", -3.0, 1.0, "all_to_all")],
    "MSd": [("MSd", -3.0, 1.0, "all_to_all")],
    "MSi": [("MSi", -3.0, 1.0, "all_to_all")],
    "FSd": [("FSd", -3.0, 1.0, "all_to_all")],
    "FSi": [("FSi", -3.0, 1.0, "all_to_all")],
    "SNr": [("SNr", -3.0, 1.0, "all_to_all")],
    "SC":  [("SC", 3.0, 1.0, "all_to_all")]
}

def connect_cross(pop_dict_left, pop_dict_right, cross_dict):
    for source, conns in cross_dict.items():
        for conn in conns:
            target, weight, delay, rule = conn
            conn_dict = {"rule": rule}
            syn_spec = {"weight": weight, "delay": delay}
            nest.Connect(pop_dict_left[source], pop_dict_right[target], conn_dict, syn_spec)
            nest.Connect(pop_dict_right[source], pop_dict_left[target], conn_dict, syn_spec)

connect_cross(LEFT, RIGHT, cross_connections)

# 7. Стартиране на симулацията
nest.Simulate(simulation_time)

# 8. Извличане на данните от spike_recorders и визуализация
def extract_raster_data(spike_rec_dict):
    data = {}
    counts = {}
    for n_type in neuron_types:
        events = nest.GetStatus(spike_rec_dict[n_type], "events")[0]
        data[n_type] = (events["times"], events["senders"])
        counts[n_type] = len(events["times"])
    return data, counts

raster_data_LEFT, spike_counts_LEFT = extract_raster_data(spike_recorders_LEFT)
raster_data_RIGHT, spike_counts_RIGHT = extract_raster_data(spike_recorders_RIGHT)

# Обединяване на спайковете от Лявата и Дясната половина
total_spike_counts = {}
for n_type in neuron_types:
    total_spike_counts[n_type] = spike_counts_LEFT[n_type] + spike_counts_RIGHT[n_type]

# Задаване на цветова схема за графиките
colors = {
    "STN": "blue",
    "GPe": "green",
    "MSd": "red",
    "MSi": "magenta",
    "FSd": "orange",
    "FSi": "purple",
    "SNr": "brown",
    "SC":  "cyan"
}

# 8.1. Raster plot – комбиниране на Лявата и Дясната половина с изместване по Y
plt.figure(figsize=(14, 8))
offset = 0
yticks = []
ytick_labels = []
for side_data, side_label in zip([raster_data_LEFT, raster_data_RIGHT], ["L", "R"]):
    for n_type in neuron_types:
        times, senders = side_data[n_type]
        if len(senders) > 0:
            shifted_senders = senders + offset
            plt.scatter(times, shifted_senders, s=10, color=colors[n_type],
                        label=f"{n_type}_{side_label}" if offset==0 else "")
        yticks.append(offset + num_neurons[n_type] / 2)
        ytick_labels.append(f"{n_type}_{side_label}")
        offset += num_neurons[n_type] + 5
plt.xlabel("Време (ms)")
plt.ylabel("ID на неврона (с изместване)")
plt.title("Raster Plot за Лява и Дясна половина")
plt.yticks(yticks, ytick_labels)
plt.legend(loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# 8.2. Bar plot – Общ брой спайкове по невронен тип (Л+Д)
plt.figure(figsize=(8, 6))
pop_names = list(total_spike_counts.keys())
spike_vals = [total_spike_counts[name] for name in pop_names]
plt.bar(pop_names, spike_vals, color=[colors[name] for name in pop_names])
plt.xlabel("Тип неврон")
plt.ylabel("Общ брой спайкове (Л+Д)")
plt.title("Общ брой спайкове по популация")
plt.tight_layout()
plt.show()

# 8.3. Хистограма на междуспайковите интервали (ISI)
plt.figure(figsize=(10, 6))
for side_data in [raster_data_LEFT, raster_data_RIGHT]:
    for n_type in neuron_types:
        times, _ = side_data[n_type]
        if len(times) > 1:
            isi = np.diff(np.sort(times))
            plt.hist(isi, bins=30, alpha=0.5, label=n_type, color=colors[n_type])
plt.xlabel("ISI (ms)")
plt.ylabel("Честота")
plt.title("Хистограма на междуспайковите интервали (ISI)")
plt.legend()
plt.tight_layout()
plt.show()

# 8.4. Плот за spike firing rate за всяка група (невронен тип)
# Изчисляваме средната firing rate (в Hz) като разделяме общия брой спайкове
# на броя неврони и времето на симулация (в секунди).
simulation_time_s = simulation_time / 1000.0  # време в секунди

firing_rates = {}
for n_type in neuron_types:
    # Средна firing rate (Hz) = (броят спайкове) / (брой неврони * време в секунди)
    firing_rates[n_type] = total_spike_counts[n_type] / (num_neurons[n_type] * simulation_time_s)

plt.figure(figsize=(8, 6))
plt.bar(list(firing_rates.keys()), list(firing_rates.values()),
        color=[colors[name] for name in firing_rates.keys()])
plt.xlabel("Тип неврон")
plt.ylabel("Spike Firing Rate (Hz)")
plt.title("Средна Spike Firing Rate по популация")
plt.tight_layout()
plt.show()


# Извличане на данните за популацията "STN" от лявата половина
stn_times, stn_senders = raster_data_LEFT["STN"]

# Плотиране на Raster Plot само за "STN"
plt.figure(figsize=(10, 4))
plt.scatter(stn_times, stn_senders, s=10, color='blue')
plt.xlabel("Време (ms)")
plt.ylabel("ID на неврона")
plt.title("Raster Plot за популацията STN (лява половина)")
plt.legend()
plt.tight_layout()
plt.show()


# Предполагаме, че сте създали популацията LEFT["STN"] и сте я свързали със spike_recorder-а.
# Нека вземем идентификатора на първия неврон от популацията LEFT["STN"]:
neuron_id = LEFT["STN"][0]

# Извличаме данните от spike_recorder-а за популацията STN в лявата половина:
stn_events = nest.GetStatus(spike_recorders_LEFT["STN"], "events")[0]

# Филтрираме спайковете само за неврона с id, равен на neuron_id
mask = stn_events["senders"] == neuron_id
neuron_times = stn_events["times"][mask]

# Сега можем да плотираме Raster Plot само за този конкретен неврон:

plt.figure(figsize=(8, 3))
plt.scatter(neuron_times, np.full_like(neuron_times, neuron_id), s=20, color='blue')
plt.xlabel("Време (ms)")
plt.ylabel("ID на неврона")
plt.title(f"Raster Plot за неврон с ID {neuron_id} от популацията STN (LEFT)")
plt.tight_layout()
plt.show()