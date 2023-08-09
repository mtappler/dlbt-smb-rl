import os
import random
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

import aalpy.paths
from aalpy.utils import mdp_2_prism_format

def extract_coords(output):
    split_o = output.replace("pos_", "").split("_")
    return int(split_o[0]),int(split_o[1])


class Scheduler:
    def __init__(self, initial_state, transition_dict, label_dict, scheduler_dict):
        self.scheduler_dict = scheduler_dict
        self.initial_state = initial_state
        self.transition_dict = transition_dict
        self.label_dict = label_dict
        self.current_state = None
        self.all_label_coords = [(extract_coords(self.get_pos_outout(o)),o,s) for s,o in self.label_dict.items() if
                                            "Init" not in o]
 
        
    def get_input(self):
        if self.current_state is None:
            print("Return none because current state is none")
            return None
        else:
            # print("Current state is not none")
            if self.current_state not in self.scheduler_dict:
                return None
            return self.scheduler_dict[self.current_state]

    def get_pos_outout(self,o_set):
        for o in o_set:
            if "pos" in o:
                return o

    def reset(self):
        self.current_state = self.initial_state

    def poss_step_to(self, input):
        output_labels = []
        trans_from_current = self.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:           
            if action == input:
                output_labels.extend(self.label_dict[target_state])
        return output_labels

    
    def step_to_closest(self, input, output):
        
        # poss_outputs = self.poss_step_to(input)
        target_coord = extract_coords(output)
        (x,y) = target_coord
        # poss_coords = [(self.extract_coords(o),o) for o in poss_outputs]
        # min_dist = 10**10
        # min_o = None
        # for (c,o) in poss_coords:
        #     (x_p,y_p) = c
        #     dist = (x_p-x)** +(y_p - y)**2
        #     if dist < min_dist:
        #         min_dist = dist
        #         min_o = o
        poss_coords = self.all_label_coords
        min_dist = 10**10
        min_o = None
        min_s = None
        for (c,o,s) in poss_coords:
            (x_p,y_p) = c
            dist = (x_p-x)**2 +(y_p - y)**2
            if dist < min_dist and s in self.scheduler_dict:
                min_dist = dist
                min_o = o
                min_s = s
        #print(self.current_state)
        self.current_state = min_s
        
        #print(self.current_state)
        #print(f"Moving to {min_o} instead of {output}")
        #return self.step_to(input,min_o)
        
        
    def step_to(self, input, output):
        reached_state = None
        trans_from_current = self.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:
            if action == input and output in self.label_dict[target_state]:
                reached_state = self.current_state = target_state
                found_state = True
                break
        if not found_state:
            return None

        return reached_state

    def get_available_actions(self):
        trans_from_current = self.transition_dict[self.current_state]
        return list(set([action for prob, action, target_state in trans_from_current]))



class PrismInterface:
    def __init__(self, destination, model, num_steps=None, maximize=True):
        self.tmp_dir = Path("tmp_prism")
        self.destination = destination
        self.model = model
        self.num_steps = num_steps
        self.maximize = maximize
        if type(destination) != list:
            destination = [destination]
        destination = "_or_".join(destination)
        self.tmp_mdp_file = (self.tmp_dir / f"po_rl_{destination}.prism")
        self.current_state = None
        self.tmp_dir.mkdir(exist_ok=True)
        self.prism_property = self.create_mc_query()
        mdp_2_prism_format(self.model, "porl", output_path=self.tmp_mdp_file)

        self.adv_file_name = (self.tmp_dir.absolute() / f"sched_{destination}.adv")
        self.concrete_model_name = str(self.tmp_dir.absolute() / f"concrete_model_{destination}")
        self.property_val = 0
        self.call_prism()
        if os.path.exists(self.adv_file_name):
            self.parser = PrismSchedulerParser(self.adv_file_name, self.concrete_model_name + ".lab",
                                               self.concrete_model_name + ".tra")
            self.scheduler = Scheduler(self.parser.initial_state, self.parser.transition_dict,
                                       self.parser.label_dict, self.parser.scheduler_dict)
            #os.remove(self.tmp_mdp_file)
            #os.remove(self.adv_file_name)
            #os.remove(self.concrete_model_name + ".lab")
            #os.remove(self.concrete_model_name + ".tra")
        else:
            self.scheduler = None

    def create_mc_query(self):
        if type(self.destination) != list:
            destination = [self.destination]
        else:
            destination = self.destination
        destination = "|".join(map(lambda d: f"\"{d}\"", destination))
        opt_string = "Pmax" if self.maximize else "Pmin"
        prop = f"{opt_string}=?[F {destination}]" if not self.num_steps else \
            f'{opt_string}=?[F<{self.num_steps} {destination}]'
        return prop

    def call_prism(self):
        import subprocess
        from os import path

        self.property_val = 0

        destination_in_model = False
        for s in self.model.states:
            if self.destination in s.output.split("__"):
                destination_in_model = True
                break

        prism_file = aalpy.paths.path_to_prism.split('/')[-1]
        path_to_prism_file = aalpy.paths.path_to_prism[:-len(prism_file)]
        file_abs_path = path.abspath(self.tmp_mdp_file)
        proc = subprocess.Popen(
            [aalpy.paths.path_to_prism, file_abs_path, "-pf", self.prism_property, "-noprob1", "-exportadvmdp",
             self.adv_file_name, "-exportmodel", f"{self.concrete_model_name}.all"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file)
        out = proc.communicate()[0]
        out = out.decode('utf-8').splitlines()
        for line in out:
            # print(line)
            if not line:
                continue
            if 'Syntax error' in line:
                print(line)
            else:
                if "Result:" in line:
                    end_index = len(line) if "(" not in line else line.index("(") - 1
                    try:
                        self.property_val = float(line[len("Result: "): end_index])
                    except:
                        print("Result parsing error")
        proc.kill()
        return self.property_val


class PrismSchedulerParser:
    def __init__(self, scheduler_file, label_file, transition_file):
        with open(scheduler_file, "r") as f:
            self.scheduler_file_content = f.readlines()
        with open(label_file, "r") as f:
            self.label_file_content = f.readlines()
        with open(transition_file, "r") as f:
            self.transition_file_content = f.readlines()
        self.label_dict = self.create_labels()
        self.transition_dict = self.create_transitions()
        self.scheduler_dict = self.parse_scheduler()
        self.initial_state = next(filter(lambda e: "init" in e[1], self.label_dict.items()))[0]
        self.actions = set()
        for l in self.transition_dict.values():
            for _, action, _ in l:
                self.actions.add(action)
        self.actions = list(self.actions)

    def create_labels(self):
        label_dict = dict()
        header_line = self.label_file_content[0]
        label_lines = self.label_file_content[1:]
        header_dict = dict()
        split_header = header_line.split(" ")
        for s in split_header:
            label_id = s.strip().split("=")[0]
            label_name = s.strip().split("=")[1].replace('"', '')
            header_dict[label_id] = label_name
        for l in label_lines:
            state_id = int(l.split(":")[0])
            label_ids = l.split(":")[1].split(" ")
            label_names = set(
                map(lambda l_id: header_dict[l_id.strip()], filter(lambda l_id: l_id.strip(), label_ids)))
            label_dict[state_id] = label_names
        return label_dict

    def create_transitions(self):
        header_line = self.transition_file_content[0]
        transition_lines = self.transition_file_content[1:]
        transitions = defaultdict(list)
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            target_state = int(split_line[2])
            prob = float(split_line[3])
            action = split_line[4].strip()
            transitions[source_state].append((prob, action, target_state))
        return transitions

    def parse_scheduler(self):
        header_line = self.scheduler_file_content[0]
        transition_lines = self.scheduler_file_content[1:]
        scheduler = dict()
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            action = split_line[4].strip()
            if source_state in scheduler:
                assert action == scheduler[source_state]
            else:
                scheduler[source_state] = action
        return scheduler

